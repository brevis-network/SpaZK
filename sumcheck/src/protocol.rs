//! Interactive Proof Protocol used for Multilinear Sumcheck

use ark_std::marker::PhantomData;
use pcs::field::JoltField;

pub mod prover;
pub mod verifier;
pub use crate::data_structures::{ListOfProductsOfPolynomials, PolynomialInfo};
/// Interactive Proof for Multilinear Sumcheck
pub struct IPForMLSumcheck<F: JoltField> {
    #[doc(hidden)]
    _marker: PhantomData<F>,
}
