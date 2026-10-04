//! The README's quick start: generate a key pair, sample a short preimage of a target, verify it.
//!
//! Usage: `cargo run --release --example quickstart`

use ntru_trapdoor::{Error, ExpandedKey, Params, RqPoly, samp_pre, trapgen, verify};

fn main() -> Result<(), Error> {
    let params = Params::PCS;
    let (pk, sk) = trapgen(&params, &[7u8; 32])?;
    // The sampling tree, built once per key and shared by every call.
    let ek = ExpandedKey::new(&sk)?;
    // Any target in R_q.
    let t = RqPoly::from_coeffs(&params, &vec![12345; params.n()])?;
    // A short s = (s0, s1) with s0 + h·s1 = t mod q and ‖s‖² ≤ β².
    let s = samp_pre(&ek, &t, &[9u8; 32])?;
    assert!(verify(&pk, &t, &s, params.beta_sq));
    println!("sampled and verified a preimage at the PCS set");
    Ok(())
}
