use thiserror::Error;

#[derive(Debug, Error)]
pub enum ZKMLError {
    /// Invalid lookup witness
    #[error("Invalid lookup witness: {0}")]
    InvalidLookupWitness(&'static str),
    /// Invalid lookup proof
    #[error("Invalid lookup claim: {0}")]
    InvalidLookupClaim(&'static str),
    /// Invalid node witness
    #[error("Invalid node witness: {0}")]
    InvalidNodeWitness(&'static str),
    /// Invalid node proof
    #[error("{0}: Wrong number of claims, want {1}, got {2}")]
    InvalidLayerClaims(&'static str, usize, usize),
    /// Invalid node proof
    #[error("Invalid node proof: {0}")]
    InvalidNodeProof(&'static str),
    /// PCS verification failed
    #[error("PCS verification failed: {0}")]
    PCSError(#[from] pcs::utils::errors::ProofVerifyError),
    /// Sumcheck verification error
    #[error("sumcheck verification failed: {0}")]
    SumcheckError(#[from] sumcheck::error::Error),
    /// Sumcheck claim verification error
    #[error("sumcheck claim not match: {0}")]
    SumcheckClaim(&'static str),
    /// Tract error
    #[error("[tract] {0}")]
    TractError(#[from] tract_onnx::prelude::TractError),
}
