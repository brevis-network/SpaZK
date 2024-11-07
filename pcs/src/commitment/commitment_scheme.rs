use std::fmt::Debug;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{
    field::JoltField,
    poly::dense::DenseMultilinearExtension,
    utils::{
        errors::ProofVerifyError,
        transcript::{AppendToTranscript, ProofTranscript},
    },
};

#[derive(Clone, Debug)]
pub struct CommitShape {
    pub input_length: usize,
    pub batch_type: BatchType,
}

impl CommitShape {
    pub fn new(input_length: usize, batch_type: BatchType) -> Self {
        Self {
            input_length,
            batch_type,
        }
    }
}

#[derive(Clone, Debug)]
pub enum BatchType {
    Big,
    Small,
    SurgeInitFinal,
    SurgeReadWrite,
}

pub trait CommitmentScheme: Clone + Sync + Send + 'static {
    type Field: JoltField + Sized;
    type Setup: Clone + Sync + Send + Debug;
    type Commitment: Sync + Send + CanonicalSerialize + CanonicalDeserialize + AppendToTranscript;
    type Proof: Sync + Send + CanonicalSerialize + CanonicalDeserialize;
    type BatchedProof: Sync + Send + CanonicalSerialize + CanonicalDeserialize;

    fn setup(shapes: &[CommitShape]) -> Self::Setup;
    fn commit(
        poly: &DenseMultilinearExtension<Self::Field>,
        setup: &Self::Setup,
    ) -> Self::Commitment;
    fn batch_commit(
        evals: &[&[Self::Field]],
        gens: &Self::Setup,
        batch_type: BatchType,
    ) -> Vec<Self::Commitment>;
    fn commit_slice(evals: &[Self::Field], setup: &Self::Setup) -> Self::Commitment;
    fn batch_commit_polys(
        polys: &[DenseMultilinearExtension<Self::Field>],
        setup: &Self::Setup,
        batch_type: BatchType,
    ) -> Vec<Self::Commitment> {
        let slices: Vec<&[Self::Field]> =
            polys.iter().map(|poly| poly.evaluations.as_ref()).collect();
        Self::batch_commit(&slices, setup, batch_type)
    }
    fn batch_commit_polys_ref(
        polys: &[&DenseMultilinearExtension<Self::Field>],
        setup: &Self::Setup,
        batch_type: BatchType,
    ) -> Vec<Self::Commitment> {
        let slices: Vec<&[Self::Field]> =
            polys.iter().map(|poly| poly.evaluations.as_ref()).collect();
        Self::batch_commit(&slices, setup, batch_type)
    }
    fn prove(
        setup: &Self::Setup,
        poly: &DenseMultilinearExtension<Self::Field>,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        transcript: &mut ProofTranscript,
    ) -> Self::Proof;
    fn batch_prove(
        setup: &Self::Setup,
        polynomials: &[&DenseMultilinearExtension<Self::Field>],
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        batch_type: BatchType,
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof;

    fn verify(
        proof: &Self::Proof,
        setup: &Self::Setup,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        opening: &Self::Field,         // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError>;

    fn batch_verify(
        batch_proof: &Self::BatchedProof,
        setup: &Self::Setup,
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        commitments: &[&Self::Commitment],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError>;

    fn protocol_name() -> &'static [u8];
}
