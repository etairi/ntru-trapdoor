//! The expanded key: the FFT basis and the ffLDL tree, built once per key; ffSampling over it.
//!
//! Port of fn-dsa-sign (Pornin, The Unlicense), commit 0629bb1:
//! `compute_basis_inner` lib.rs:456-477 (basis $`B=[[g,-f],[G,-F]]`$ in FFT form), the Gram
//! matrix of lib.rs:575-616 (precomputed-basis branch), and the LDL and split steps of
//! `ffsamp_fft_inner` (sampler.rs:424-635), moved from signing time to key-expansion time
//! (Falcon's Alg. 9/11 layout; falcon-rs src/sign.rs:22-609 is a layout reference only).
//!
//! The tree builder performs fn-dsa's on-the-fly computation (the same IEEE operations in the
//! same order, design.md §5.4), so that the tree holds, bit for bit, the values that fn-dsa
//! computes during each signature. The unit test `tree_equals_per_attempt_ldl` checks this
//! against a copy of fn-dsa's per-attempt recursion (LDL and splits during sampling) over the
//! same FFT primitives: every leaf centre and width agrees bit for bit. The parity tests against
//! fn-dsa itself compare outputs; they certify every discrete decision and the constants to about
//! $`10^{-8}`$ relative, not IEEE bit patterns (fn-dsa parity tests):
//! - node at `logn >= 2`: `poly_ldl_fft(g00, g01, g11)`; store `l10` (n values); split `d00`
//!   and `d11` with `poly_split_selfadj_fft`; children: right = Gram of the split of `d11`
//!   (sampled first, for `t1`), left = Gram of the split of `d00`;
//! - node at `logn = 1`: fn-dsa's inline LDL (sampler.rs:506-557); store `l01 = (mu_re,
//!   -mu_im)` and the two leaf values `sqrt(d11) * inv_sigma` and `sqrt(d00) * inv_sigma`
//!   (that is $`1/\sigma'`$).
//!
//! Layout: depth-first in sampling order, `[l10 | right subtree | left subtree]`, size
//! $`(\mathrm{logn}+1)\,2^{\mathrm{logn}}`$ values (11,264 `f64` = 90,112 B at $`d=1024`$; with the
//! 4n-value basis the expanded key is 122,880 B).
//!
//! Leaf check (rejects with [`Error::LeafOutOfRange`]): for every leaf value $`v=1/\sigma'`$,
//! `v * sigma_min <= 1.0` (the very expression of the sampler's `ccs`, so $`ccs\le1`$) and
//! `(v * v) * 0.5 >= inv_2sqr_sigma0 * (1 + 2^-40)` (the sampler's `dss`; the margin keeps the
//! Bernoulli exponent nonnegative after rounding, design.md §5.4). fn-dsa checks neither.
//!
//! ffSampling (`ffsampling`) mirrors fn-dsa's `ffsamp_fft_inner` (sampler.rs:505-635) with tree
//! lookups instead of the LDL: leaf order at each `logn = 1` node is `t1[0], t1[1]`, then
//! `t0[0], t0[1]` (fn-dsa's order). The product `(t1 - z1) * l10` is computed as
//! `poly_mul_fft(w, l10)` instead of fn-dsa's `poly_mul_fft(l10, w)`: the same two IEEE products
//! per component, combined by the same operation (a commuted addition for the imaginary part),
//! hence bit-identical.
//!
//! [`ExpandedKey::new`] first runs [`SecretKey::validate`] unless the key is known valid (it
//! came from key generation or [`SecretKey::from_bytes`]): an inconsistent trapdoor
//! ($`fG-gF\ne q`$) would otherwise give "preimages" with $`A\,s\ne t`$ that expose the sampler's
//! integer coordinates.
//!
//! Used by key generation and the tests: [`ExpandedKey::new`],
//! [`ExpandedKey::params`], [`ExpandedKey::leaf_sigma_range`].

use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::fft::{
    fft, poly_add, poly_ldl_fft, poly_merge_fft, poly_mul_fft, poly_muladj_fft, poly_mulownadj_fft,
    poly_neg, poly_set_small, poly_split_fft, poly_split_selfadj_fft, poly_sub,
};
use crate::prng::shake256;
use crate::sampler::LeafSampler;
use crate::{Error, Params, SecretKey};

/// Domain-separation label of the key identifier (design.md §5.1).
const KEY_ID_LABEL: &[u8] = b"ntru-trapdoor/v1/key-id";

/// The relative rounding margin of the leaf check on the base-width side: $`2^{-40}`$.
const LEAF_MARGIN: f64 = 1.0 + 1.0 / (1u64 << 40) as f64;

/// A secret key expanded for sampling: FFT basis, ffLDL tree, and a key identifier for the
/// hedged seed derivation. Read-only after construction (`Send + Sync`); wiped on drop.
pub struct ExpandedKey {
    pub(crate) params: Params,
    /// `[b00 | b01 | b10 | b11]` = FFT of `[g, -f, G, -F]`, `4n` values.
    pub(crate) basis: Vec<f64>,
    /// The ffLDL tree, `(logn + 1) * n` values.
    pub(crate) tree: Vec<f64>,
    /// SHAKE256 of the key, for the hedged per-call seeds (design.md §5.7).
    pub(crate) key_id: [u8; 32],
    /// The smallest and largest leaf width $`\sigma'_i`$.
    leaf_min: f64,
    leaf_max: f64,
}

impl Zeroize for ExpandedKey {
    /// Wipes the basis, the tree, the key identifier and the leaf range, and truncates the
    /// vectors. Slice wipes (one volatile pass each): `Vec::zeroize` would also wipe the whole
    /// capacity byte by byte, about ten times the cost [measured, sign/README.md]; the vectors are
    /// allocated at their exact size, so the slices cover their capacity.
    fn zeroize(&mut self) {
        self.basis.as_mut_slice().zeroize();
        self.basis.clear();
        self.tree.as_mut_slice().zeroize();
        self.tree.clear();
        self.key_id.zeroize();
        self.leaf_min.zeroize();
        self.leaf_max.zeroize();
    }
}

impl Drop for ExpandedKey {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl ZeroizeOnDrop for ExpandedKey {}

/// Number of `f64` values of the tree of a node at `logn`: $`(\mathrm{logn}+1)2^{\mathrm{logn}}`$.
pub(crate) const fn tree_size(logn: u32) -> usize {
    (logn as usize + 1) << logn
}

impl ExpandedKey {
    /// Builds the basis, the Gram matrix and the tree; checks every leaf width against the base
    /// sampler's range. A key from [`SecretKey::from_parts`] is first checked with
    /// [`SecretKey::validate`] (about 70 µs at $`d=1024`$); keys from [`crate::trapgen`] and
    /// [`SecretKey::from_bytes`] were validated already. Errors: [`Error::InvalidKey`],
    /// [`Error::LeafOutOfRange`].
    pub fn new(sk: &SecretKey) -> Result<ExpandedKey, Error> {
        if !sk.validated {
            sk.validate()?;
        }
        ExpandedKey::from_valid(sk)
    }

    /// [`ExpandedKey::new`] for a key that passed [`SecretKey::validate`] (key generation calls
    /// it for its leaf check).
    pub(crate) fn from_valid(sk: &SecretKey) -> Result<ExpandedKey, Error> {
        // sk.params passed Params::validate when the key was assembled.
        let params = sk.params;
        let logn = params.logn;
        let n = params.n();

        // Basis B = [[g, -f], [G, -F]] in FFT form (fn-dsa lib.rs:456-477).
        let mut basis = vec![0.0f64; 4 * n];
        {
            let (b00, rest) = basis.split_at_mut(n);
            let (b01, rest) = rest.split_at_mut(n);
            let (b10, b11) = rest.split_at_mut(n);
            poly_set_small(logn, b01, &sk.f);
            poly_set_small(logn, b00, &sk.g);
            poly_set_small(logn, b11, &sk.big_f);
            poly_set_small(logn, b10, &sk.big_g);
            fft(logn, b01);
            fft(logn, b00);
            fft(logn, b11);
            fft(logn, b10);
            poly_neg(logn, b01);
            poly_neg(logn, b11);
        }

        // Gram matrix G = B adj(B) (fn-dsa lib.rs:575-616, precomputed-basis branch):
        // g00 = b00 adj(b00) + b01 adj(b01), g01 = b00 adj(b10) + b01 adj(b11),
        // g11 = b10 adj(b10) + b11 adj(b11).
        let mut work = vec![0.0f64; 6 * n];
        {
            let (b00, rest) = basis.split_at(n);
            let (b01, rest) = rest.split_at(n);
            let (b10, b11) = rest.split_at(n);
            let (g00, rest) = work.split_at_mut(n);
            let (g01, rest) = rest.split_at_mut(n);
            let (g11, rest) = rest.split_at_mut(n);
            let (t0, tmp) = rest.split_at_mut(n);

            g00.copy_from_slice(b00);
            poly_mulownadj_fft(logn, g00);
            t0.copy_from_slice(b01);
            poly_mulownadj_fft(logn, t0);
            poly_add(logn, g00, t0);

            g01.copy_from_slice(b00);
            poly_muladj_fft(logn, g01, b10);
            t0.copy_from_slice(b01);
            poly_muladj_fft(logn, t0, b11);
            poly_add(logn, g01, t0);

            g11.copy_from_slice(b10);
            poly_mulownadj_fft(logn, g11);
            t0.copy_from_slice(b11);
            poly_mulownadj_fft(logn, t0);
            poly_add(logn, g11, t0);

            // The tree (the LDL and split steps of fn-dsa's ffsamp_fft_inner).
            let mut tree = vec![0.0f64; tree_size(logn)];
            build_tree(logn, g00, g01, g11, &mut tree, tmp, params.inv_sigma);
            work.as_mut_slice().zeroize();

            // Leaf check.
            let sigma_min = params.sigma_min;
            let dss_min = params.base.inv_2sqr_sigma0() * LEAF_MARGIN;
            let mut ok = true;
            let (mut vmin, mut vmax) = (f64::INFINITY, 0.0f64);
            for_each_leaf(logn, &tree, &mut |v| {
                ok &= v * sigma_min <= 1.0;
                ok &= (v * v) * 0.5 >= dss_min;
                vmin = vmin.min(v);
                vmax = vmax.max(v);
            });
            if !ok || !(vmin > 0.0 && vmax.is_finite()) {
                tree.as_mut_slice().zeroize();
                basis.as_mut_slice().zeroize();
                return Err(Error::LeafOutOfRange);
            }

            // Key identifier (design.md §5.1).
            let mut enc = Vec::with_capacity(8 * n);
            for poly in [&sk.f, &sk.g, &sk.big_f, &sk.big_g] {
                for &c in poly.iter() {
                    enc.extend_from_slice(&c.to_le_bytes());
                }
            }
            let mut key_id = [0u8; 32];
            shake256(
                &[KEY_ID_LABEL, &params.q.to_le_bytes(), &[logn as u8], &enc],
                &mut key_id,
            );
            enc.as_mut_slice().zeroize();

            Ok(ExpandedKey {
                params,
                basis,
                tree,
                key_id,
                leaf_min: 1.0 / vmax,
                leaf_max: 1.0 / vmin,
            })
        }
    }

    /// The parameter set.
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// The smallest and the largest leaf width $`\sigma'_i`$ over the $`2d`$ leaves.
    pub fn leaf_sigma_range(&self) -> (f64, f64) {
        (self.leaf_min, self.leaf_max)
    }

    /// The four FFT basis polynomials `(b00, b01, b10, b11)`.
    pub(crate) fn basis_parts(&self) -> (&[f64], &[f64], &[f64], &[f64]) {
        let n = self.params.n();
        let (b00, rest) = self.basis.split_at(n);
        let (b01, rest) = rest.split_at(n);
        let (b10, b11) = rest.split_at(n);
        (b00, b01, b10, b11)
    }
}

/// Builds the tree of the Gram matrix `[[g00, g01], [adj(g01), g11]]` (FFT representation,
/// `g00` and `g11` self-adjoint) into `tree` (`tree_size(logn)` values). Clobbers the Gram
/// matrix; `tmp` needs `n` values. This is fn-dsa's `ffsamp_fft_inner` (sampler.rs:505-635)
/// without the sampling.
fn build_tree(
    logn: u32,
    g00: &mut [f64],
    g01: &mut [f64],
    g11: &mut [f64],
    tree: &mut [f64],
    tmp: &mut [f64],
    inv_sigma: f64,
) {
    if logn == 1 {
        // fn-dsa sampler.rs:506-518: inline LDL of the 2x2 matrix of complex numbers.
        let g00_re = g00[0];
        let (g01_re, g01_im) = (g01[0], g01[1]);
        let g11_re = g11[0];
        let inv_g00_re = 1.0 / g00_re;
        let (mu_re, mu_im) = (g01_re * inv_g00_re, g01_im * inv_g00_re);
        let zo_re = mu_re * g01_re + mu_im * g01_im;
        let d00_re = g00_re;
        let d11_re = g11_re - zo_re;
        tree[0] = mu_re;
        tree[1] = -mu_im;
        // Leaf values 1/sigma' (sampler.rs:532, :553): right leaf (t1) first, then left (t0).
        tree[2] = d11_re.sqrt() * inv_sigma;
        tree[3] = d00_re.sqrt() * inv_sigma;
        return;
    }

    let n = 1usize << logn;
    let hn = n >> 1;

    // Decompose G into LDL; g01 <- l10, g11 <- d11 (sampler.rs:565).
    poly_ldl_fft(logn, g00, g01, g11);

    // Split d00 and d11 into half-size quasi-cyclic Gram matrices (sampler.rs:570-579).
    {
        let (w0, w1) = tmp[..n].split_at_mut(hn);
        poly_split_selfadj_fft(logn, w0, w1, g00);
        g00[..hn].copy_from_slice(w0);
        g00[hn..n].copy_from_slice(w1);
        poly_split_selfadj_fft(logn, w0, w1, g11);
        g11[..hn].copy_from_slice(w0);
        g11[hn..n].copy_from_slice(w1);
    }
    // Store l10, then (sampler.rs:580-582):
    //   left sub-tree:  g00[0..hn], g00[hn..n], g01[0..hn]
    //   right sub-tree: g11[0..hn], g11[hn..n], g01[hn..n]
    tree[..n].copy_from_slice(&g01[..n]);
    g01[..hn].copy_from_slice(&g00[..hn]);
    g01[hn..n].copy_from_slice(&g11[..hn]);

    let sub = tree_size(logn - 1);
    let (_, rest) = tree.split_at_mut(n);
    let (right, left) = rest.split_at_mut(sub);
    let (left_00, left_01) = g00[..n].split_at_mut(hn);
    let (right_00, right_01) = g11[..n].split_at_mut(hn);
    let (left_11, right_11) = g01[..n].split_at_mut(hn);
    build_tree(
        logn - 1,
        right_00,
        right_01,
        right_11,
        right,
        tmp,
        inv_sigma,
    );
    build_tree(logn - 1, left_00, left_01, left_11, left, tmp, inv_sigma);
}

/// Visits the leaf values in sampling order (right leaf before left leaf at each `logn = 1`
/// node, right subtree before left subtree).
fn for_each_leaf(logn: u32, tree: &[f64], f: &mut impl FnMut(f64)) {
    if logn == 1 {
        f(tree[2]);
        f(tree[3]);
        return;
    }
    let n = 1usize << logn;
    let sub = tree_size(logn - 1);
    for_each_leaf(logn - 1, &tree[n..n + sub], f);
    for_each_leaf(logn - 1, &tree[n + sub..n + 2 * sub], f);
}

/// Fast Fourier sampling over the tree (fn-dsa `ffsamp_fft_inner`, sampler.rs:505-635, with
/// tree lookups): on input the target `(t0, t1)` (FFT representation), on output the sampled
/// `(z0, z1)` (FFT representation). `tmp` needs `2n` values.
#[inline]
pub(crate) fn ffsampling<S: LeafSampler>(
    samp: &mut S,
    logn: u32,
    t0: &mut [f64],
    t1: &mut [f64],
    tree: &[f64],
    tmp: &mut [f64],
) {
    if logn == 1 {
        // sampler.rs:527-557. t1's split is trivial: w0 = t1[0], w1 = t1[1].
        let (l01_re, l01_im, leaf_r, leaf_l) = (tree[0], tree[1], tree[2], tree[3]);
        let w0 = t1[0];
        let w1 = t1[1];
        let y0 = samp.sample(w0, leaf_r) as f64;
        let y1 = samp.sample(w1, leaf_r) as f64;
        // tb0 = t0 + (t1 - z1) * l10, z1 into t1.
        let (a_re, a_im) = (w0 - y0, w1 - y1);
        let (b_re, b_im) = (a_re * l01_re - a_im * l01_im, a_re * l01_im + a_im * l01_re);
        let (x0, x1) = (t0[0] + b_re, t0[1] + b_im);
        t1[0] = y0;
        t1[1] = y1;
        t0[0] = samp.sample(x0, leaf_l) as f64;
        t0[1] = samp.sample(x1, leaf_l) as f64;
        return;
    }

    let n = 1usize << logn;
    let hn = n >> 1;
    let sub = tree_size(logn - 1);
    let (l10, rest) = tree.split_at(n);
    let (right, left) = rest.split_at(sub);

    // Split t1, sample the right sub-tree, merge into z1 = tmp[n..2n] (sampler.rs:593-602).
    {
        let (w0, rest) = tmp.split_at_mut(hn);
        let (w1, rest) = rest.split_at_mut(hn);
        poly_split_fft(logn, w0, w1, t1);
        ffsampling(samp, logn - 1, w0, w1, right, rest);
        poly_merge_fft(logn, &mut rest[..n], w0, w1);
    }

    // tb0 = t0 + (t1 - z1) * l10 into t0; z1 into t1 (sampler.rs:604-620).
    {
        let (w, rest) = tmp.split_at_mut(n);
        let z1 = &rest[..n];
        w.copy_from_slice(&t1[..n]);
        poly_sub(logn, w, z1);
        t1[..n].copy_from_slice(z1);
        poly_mul_fft(logn, w, l10);
        poly_add(logn, t0, w);
    }

    // Split tb0, sample the left sub-tree, merge into t0 (sampler.rs:622-634).
    {
        let (w0, rest) = tmp.split_at_mut(hn);
        let (w1, rest) = rest.split_at_mut(hn);
        poly_split_fft(logn, w0, w1, t0);
        ffsampling(samp, logn - 1, w0, w1, left, rest);
        poly_merge_fft(logn, t0, w0, w1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::fft::{fft, poly_mul_fft, poly_mulconst, poly_set_small, poly_sub};
    use crate::prng::{Prng, Shake256Prng};

    /// The tree holds n leaf values d (one per pair of samples: the two samples of a `logn = 1`
    /// node's t1, or of its t0, share a width), and the 2n squared Gram–Schmidt norms of B are
    /// these values, each twice. So prod d^2 = det(B B*) = det(B)^2, i.e.
    /// sum ln d = ln det(B) = sum over the n embeddings of ln |fG - gF| (B as a 2n x 2n real
    /// matrix). Checked for random (not NTRU) bases at every degree.
    #[test]
    fn leaves_multiply_to_det_squared() {
        for logn in 1..=10u32 {
            let n = 1usize << logn;
            let hn = n / 2;
            let mut rng = Shake256Prng::new(&[b'd', logn as u8]);
            let mut small =
                || -> Vec<i16> { (0..n).map(|_| (rng.next_u8() % 21) as i16 - 10).collect() };
            let (f, g, big_f, big_g) = (small(), small(), small(), small());
            // A Params with inv_sigma = 1 so that leaf v_i = sqrt(d_i); the leaf check is not
            // used here (we call the builder directly).
            let mut basis = vec![0.0; 4 * n];
            let (b00, rest) = basis.split_at_mut(n);
            let (b01, rest) = rest.split_at_mut(n);
            let (b10, b11) = rest.split_at_mut(n);
            poly_set_small(logn, b01, &f);
            poly_set_small(logn, b00, &g);
            poly_set_small(logn, b11, &big_f);
            poly_set_small(logn, b10, &big_g);
            for b in [&mut *b00, &mut *b01, &mut *b10, &mut *b11] {
                fft(logn, b);
            }
            // det: e = f G - g F in FFT form; each stored point stands for a conjugate pair.
            let mut e = b01.to_vec();
            poly_mul_fft(logn, &mut e, b10);
            let mut t = b00.to_vec();
            poly_mul_fft(logn, &mut t, b11);
            poly_sub(logn, &mut e, &t);
            let log_det: f64 = (0..hn)
                .map(|j| 2.0 * (e[j] * e[j] + e[j + hn] * e[j + hn]).sqrt().ln())
                .sum();
            for b in [&mut *b01, &mut *b11] {
                for x in b.iter_mut() {
                    *x = -*x;
                }
            }
            // Gram matrix and tree, as ExpandedKey::new.
            let mut work = vec![0.0; 6 * n];
            let (g00, rest) = work.split_at_mut(n);
            let (g01, rest) = rest.split_at_mut(n);
            let (g11, rest) = rest.split_at_mut(n);
            let (t0, tmp) = rest.split_at_mut(n);
            g00.copy_from_slice(b00);
            poly_mulownadj_fft(logn, g00);
            t0.copy_from_slice(b01);
            poly_mulownadj_fft(logn, t0);
            poly_add(logn, g00, t0);
            g01.copy_from_slice(b00);
            poly_muladj_fft(logn, g01, b10);
            t0.copy_from_slice(b01);
            poly_muladj_fft(logn, t0, b11);
            poly_add(logn, g01, t0);
            g11.copy_from_slice(b10);
            poly_mulownadj_fft(logn, g11);
            t0.copy_from_slice(b11);
            poly_mulownadj_fft(logn, t0);
            poly_add(logn, g11, t0);
            let mut tree = vec![0.0; tree_size(logn)];
            build_tree(logn, g00, g01, g11, &mut tree, tmp, 1.0);
            let mut sum_log_d = 0.0;
            let mut count = 0;
            for_each_leaf(logn, &tree, &mut |v| {
                sum_log_d += 2.0 * v.ln();
                count += 1;
            });
            assert_eq!(count, n);
            assert!(
                (sum_log_d - log_det).abs() <= 1e-9 * sum_log_d.abs().max(1.0),
                "logn {logn}: {sum_log_d} vs {log_det}"
            );
        }
    }

    /// A recording leaf sampler: SamplerZ over a SHAKE256 stream, logging every call's inputs
    /// (as bit patterns) and output.
    struct Recorder<'a> {
        inner: crate::sampler::SamplerZ<'a, Shake256Prng>,
        log: Vec<(u64, u64, i32)>,
    }

    impl LeafSampler for Recorder<'_> {
        fn sample(&mut self, mu: f64, isigma: f64) -> i32 {
            let z = self.inner.sample(mu, isigma);
            self.log.push((mu.to_bits(), isigma.to_bits(), z));
            z
        }
    }

    /// fn-dsa's per-attempt recursion `ffsamp_fft_inner` (fn-dsa-sign/src/sampler.rs:505-635,
    /// the portable branch), with the LDL and the splits computed during sampling, over this
    /// crate's FFT primitives (themselves ports of fn-dsa's). Statement for statement as fn-dsa,
    /// including its `poly_mul_fft(l10, w)` operand order; `g00`, `g01`, `g11` are consumed.
    #[allow(clippy::too_many_arguments)]
    fn ffsamp_per_attempt<S: LeafSampler>(
        samp: &mut S,
        logn: u32,
        inv_sigma: f64,
        t0: &mut [f64],
        t1: &mut [f64],
        g00: &mut [f64],
        g01: &mut [f64],
        g11: &mut [f64],
        tmp: &mut [f64],
    ) {
        if logn == 1 {
            let g00_re = g00[0];
            let (g01_re, g01_im) = (g01[0], g01[1]);
            let g11_re = g11[0];
            let inv_g00_re = 1.0 / g00_re;
            let (mu_re, mu_im) = (g01_re * inv_g00_re, g01_im * inv_g00_re);
            let zo_re = mu_re * g01_re + mu_im * g01_im;
            let d00_re = g00_re;
            let l01_re = mu_re;
            let l01_im = -mu_im;
            let d11_re = g11_re - zo_re;
            let w0 = t1[0];
            let w1 = t1[1];
            let leaf = d11_re.sqrt() * inv_sigma;
            let y0 = samp.sample(w0, leaf) as f64;
            let y1 = samp.sample(w1, leaf) as f64;
            let (a_re, a_im) = (w0 - y0, w1 - y1);
            // flc_mul (fn-dsa-sign/src/poly.rs:34-38).
            let (b_re, b_im) = (a_re * l01_re - a_im * l01_im, a_re * l01_im + a_im * l01_re);
            let (x0, x1) = (t0[0] + b_re, t0[1] + b_im);
            t1[0] = y0;
            t1[1] = y1;
            let leaf = d00_re.sqrt() * inv_sigma;
            t0[0] = samp.sample(x0, leaf) as f64;
            t0[1] = samp.sample(x1, leaf) as f64;
            return;
        }
        let n = 1usize << logn;
        let hn = n >> 1;
        poly_ldl_fft(logn, g00, g01, g11);
        {
            let (w0, w1) = tmp[..n].split_at_mut(hn);
            poly_split_selfadj_fft(logn, w0, w1, g00);
            g00[..hn].copy_from_slice(w0);
            g00[hn..n].copy_from_slice(w1);
            poly_split_selfadj_fft(logn, w0, w1, g11);
            g11[..hn].copy_from_slice(w0);
            g11[hn..n].copy_from_slice(w1);
        }
        tmp[..n].copy_from_slice(&g01[..n]);
        g01[..hn].copy_from_slice(&g00[..hn]);
        g01[hn..n].copy_from_slice(&g11[..hn]);
        let (left_00, left_01) = g00[..n].split_at_mut(hn);
        let (right_00, right_01) = g11[..n].split_at_mut(hn);
        let (left_11, right_11) = g01[..n].split_at_mut(hn);
        {
            let (_, tmp) = tmp.split_at_mut(n);
            let (w0, tmp) = tmp.split_at_mut(hn);
            let (w1, tmp) = tmp.split_at_mut(hn);
            poly_split_fft(logn, w0, w1, t1);
            ffsamp_per_attempt(
                samp,
                logn - 1,
                inv_sigma,
                w0,
                w1,
                right_00,
                right_01,
                right_11,
                tmp,
            );
            poly_merge_fft(logn, &mut tmp[..n], w0, w1);
        }
        {
            let (l10, tmp) = tmp.split_at_mut(n);
            let (w, z1) = tmp.split_at_mut(n);
            w.copy_from_slice(&t1[..n]);
            poly_sub(logn, w, &z1[..n]);
            t1[..n].copy_from_slice(&z1[..n]);
            poly_mul_fft(logn, l10, w);
            poly_add(logn, t0, l10);
        }
        {
            let (w0, tmp) = tmp.split_at_mut(hn);
            let (w1, tmp) = tmp.split_at_mut(hn);
            poly_split_fft(logn, w0, w1, t0);
            ffsamp_per_attempt(
                samp,
                logn - 1,
                inv_sigma,
                w0,
                w1,
                left_00,
                left_01,
                left_11,
                tmp,
            );
            poly_merge_fft(logn, t0, w0, w1);
        }
    }

    /// Design claim D1: the precomputed tree holds, bit for bit, the
    /// values of fn-dsa's per-attempt recursion. For TrapGen keys of both presets and several
    /// targets, the tree-based `ffsampling` and the per-attempt copy above, fed the same PRNG
    /// stream, make identical leaf calls (centre and inverse width compared as bit patterns) and
    /// return identical samples.
    #[test]
    fn tree_equals_per_attempt_ldl() {
        for (params, keys) in [(Params::PCS, 3u8), (Params::FALCON_1024, 3u8)] {
            let logn = params.logn;
            let n = params.n();
            for k in 0..keys {
                let (_, sk) = crate::trapgen(&params, &[k.wrapping_mul(29).wrapping_add(3); 32])
                    .expect("trapgen");
                let ek = ExpandedKey::new(&sk).expect("expand");
                let (b00, b01, b10, b11) = ek.basis_parts();
                let mut rng = Shake256Prng::from_parts(&[b"d1-targets", &[k]]);
                for case in 0..4u32 {
                    let t: Vec<u32> = match case {
                        0 => vec![0; n],
                        1 => vec![params.q - 1; n],
                        _ => (0..n)
                            .map(|_| (rng.next_u64() % params.q as u64) as u32)
                            .collect(),
                    };
                    // The target transform of samp_pre_with.
                    let mut t0: Vec<f64> = t.iter().map(|&c| c as f64).collect();
                    fft(logn, &mut t0);
                    let mut t1 = t0.clone();
                    poly_mul_fft(logn, &mut t1, b01);
                    poly_mulconst(logn, &mut t1, -params.inv_q);
                    poly_mul_fft(logn, &mut t0, b11);
                    poly_mulconst(logn, &mut t0, params.inv_q);
                    let (mut u0, mut u1) = (t0.clone(), t1.clone());
                    // The Gram matrix, as ExpandedKey::new computes it.
                    let mut g00 = b00.to_vec();
                    poly_mulownadj_fft(logn, &mut g00);
                    let mut tt = b01.to_vec();
                    poly_mulownadj_fft(logn, &mut tt);
                    poly_add(logn, &mut g00, &tt);
                    let mut g01 = b00.to_vec();
                    poly_muladj_fft(logn, &mut g01, b10);
                    let mut tt = b01.to_vec();
                    poly_muladj_fft(logn, &mut tt, b11);
                    poly_add(logn, &mut g01, &tt);
                    let mut g11 = b10.to_vec();
                    poly_mulownadj_fft(logn, &mut g11);
                    let mut tt = b11.to_vec();
                    poly_mulownadj_fft(logn, &mut tt);
                    poly_add(logn, &mut g11, &tt);

                    let seed = [b"d1-stream".as_slice(), &[k, case as u8]].concat();
                    let mut p1 = Shake256Prng::new(&seed);
                    let mut r1 = Recorder {
                        inner: crate::sampler::SamplerZ::new(&params, &mut p1),
                        log: Vec::new(),
                    };
                    let mut tmp = vec![0.0; 2 * n];
                    ffsampling(&mut r1, logn, &mut t0, &mut t1, &ek.tree, &mut tmp);
                    let mut p2 = Shake256Prng::new(&seed);
                    let mut r2 = Recorder {
                        inner: crate::sampler::SamplerZ::new(&params, &mut p2),
                        log: Vec::new(),
                    };
                    // fn-dsa's recursion needs 4n - 4 scratch values (l10, w0 | w1, z1 per level).
                    let mut tmp = vec![0.0; 4 * n];
                    ffsamp_per_attempt(
                        &mut r2,
                        logn,
                        params.inv_sigma,
                        &mut u0,
                        &mut u1,
                        &mut g00,
                        &mut g01,
                        &mut g11,
                        &mut tmp,
                    );
                    assert_eq!(r1.log.len(), 2 * n);
                    assert_eq!(r1.log, r2.log, "{} key {k} target {case}", params.name);
                    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
                    assert_eq!(bits(&t0), bits(&u0));
                    assert_eq!(bits(&t1), bits(&u1));
                }
            }
        }
    }

    /// S-ZERO-1: an explicit `zeroize()` wipes the basis, the tree, the key identifier and the
    /// leaf range (the derive also runs it on drop).
    #[test]
    fn expanded_key_zeroize() {
        let mut ek = ExpandedKey {
            params: Params::FALCON_1024,
            basis: vec![1.5; 16],
            tree: vec![2.5; 12],
            key_id: [7; 32],
            leaf_min: 1.0,
            leaf_max: 2.0,
        };
        let (bp, tp) = (ek.basis.as_ptr(), ek.tree.as_ptr());
        ek.zeroize();
        assert!(ek.basis.is_empty() && ek.tree.is_empty());
        // The vectors keep their buffers (zeroize clears the full capacity, then truncates).
        assert_eq!((ek.basis.as_ptr(), ek.tree.as_ptr()), (bp, tp));
        assert_eq!(ek.key_id, [0; 32]);
        assert_eq!((ek.leaf_min, ek.leaf_max), (0.0, 0.0));
    }

    #[test]
    fn tree_sizes() {
        assert_eq!(tree_size(1), 4);
        assert_eq!(tree_size(2), 12);
        assert_eq!(tree_size(10), 11264);
        for logn in 2..=10 {
            assert_eq!(tree_size(logn), (1 << logn) + 2 * tree_size(logn - 1));
        }
    }
}
