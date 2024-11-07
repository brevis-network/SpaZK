use std::fmt::Debug;

use pcs::{
    commitment::commitment_scheme::CommitmentScheme, field::JoltField,
    poly::dense::DenseMultilinearExtension, utils::transcript::ProofTranscript,
};

use crate::{
    error::ZKMLError,
    lookup::{LookupClaim, LookupWitness},
    tensor::{TensorClaim, TensorData, Tensors},
};

pub mod linear;
pub mod relu;
pub mod sparse_linear;
pub mod ternary_sparse_linear;

pub trait LayerGenerator<T: TensorData, PCS: CommitmentScheme> {
    fn prover_and_verifier(&self) -> (Box<dyn LayerProver<PCS>>, Box<dyn LayerVerifier<PCS>>);
    fn forward(
        &self,
        inputs: &Tensors<T>,
    ) -> (
        Tensors<T>,
        LayerWitness<PCS::Field>,
        LookupWitness<PCS::Field>,
    );
}

pub trait LayerProver<PCS: CommitmentScheme> {
    fn prove(
        &self,
        inputs: DenseMultilinearExtension<PCS::Field>,
        witness: LayerWitness<PCS::Field>,
        lo_point: &[PCS::Field],
        hi_point: &[PCS::Field],
        lookup_point: &[PCS::Field],
        transcript: &mut ProofTranscript,
    ) -> Result<LayerProverOutput<PCS>, ZKMLError>;
}

pub trait LayerVerifier<PCS: CommitmentScheme> {
    #[allow(clippy::type_complexity)]
    fn verify(
        &self,
        output_claim: &TensorClaim<PCS::Field>,
        lookup_claim: &Option<LookupClaim<PCS::Field>>,
        proof: LayerProof<PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(TensorClaim<PCS::Field>, Vec<LayerClaim<PCS::Field>>), ZKMLError>;
}

#[derive(Clone, Debug, Default)]
pub struct LayerWitness<F: JoltField>(pub Vec<DenseMultilinearExtension<F>>);

pub type SumcheckProof<F> = sumcheck::Proof<F>;

pub struct LayerProverOutput<PCS: CommitmentScheme> {
    pub proof: LayerProof<PCS>,
    pub lo_point: Vec<PCS::Field>,
    pub hi_point: Vec<PCS::Field>,
}

pub struct LayerProof<PCS: CommitmentScheme> {
    pub sc_proof: SumcheckProof<PCS::Field>,
    pub claims: Vec<PCS::Field>,
    pub model_openings: Vec<PCS::Proof>,
}

pub struct LayerClaim<F: JoltField> {
    pub lo_point: Vec<F>,
    pub hi_point: Vec<F>,
    pub value: F,
}
