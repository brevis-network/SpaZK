use pcs::field::JoltField;
use prover::ProverState;
use sumcheck::{protocol::PolynomialInfo, Proof};
use verifier::VerifierState;

use crate::circuit::{Circuit, Witness};

pub mod prover;
pub mod verifier;

pub struct GKRProof<F: JoltField> {
    pub sum_1: F,

    // phase 1
    identity_1_info: PolynomialInfo,
    proof_1: Proof<F>,
    eval_in_0: F,
    eval_h: F,
    // phase 2
    identity_2_info: PolynomialInfo,
    proof_2: Proof<F>,
    eval_in_1: F,
}

pub fn gkr_prover<F: JoltField, C: Circuit<F>>(
    label: (&'static str, usize, usize),
    circuit: &C,
    circuit_witness: Witness<F>,
    sum_1: F,
    rz: &[F],
) -> GKRProof<F> {
    let mut prover_state = ProverState::new();

    let identity_1 = prover_state.prover_phase1_setup(label, circuit, &circuit_witness, rz);
    let (proof_1, rx, eval_in_0, eval_h) = prover_state
        .prover_phase1(label, &identity_1)
        .expect("prover phase1 failed");

    let identity_2 = prover_state.prover_phase2_setup(label, circuit, &circuit_witness, &rx);
    let (proof_2, _, eval_in_1) = prover_state
        .prover_phase2(label, &identity_2)
        .expect("prover phase2 failed");
    GKRProof {
        sum_1,
        identity_1_info: identity_1.info(),
        proof_1,
        eval_in_0,
        eval_h,
        identity_2_info: identity_2.info(),
        proof_2,
        eval_in_1,
    }
}

pub fn gkr_verifier<F: JoltField, C: Circuit<F>>(
    label: (&'static str, usize, usize),
    circuit: &C,
    rz: &[F],
    proof: GKRProof<F>,
) {
    let mut verifier_state = VerifierState::new(rz.to_vec());
    let sub_claim_1 = verifier_state
        .verifier_phase1(label, &proof.identity_1_info, proof.sum_1, proof.proof_1)
        .expect("verifier phase1 sumcheck failed");

    verifier_state
        .verifier_finalize_1(label, sub_claim_1, proof.eval_in_0, proof.eval_h)
        .expect("verifier phase1 finalization failed");

    let sub_claim_2 = verifier_state
        .verifier_phase2(label, &proof.identity_2_info, proof.eval_h, proof.proof_2)
        .expect("verifier phase2 sumcheck failed");
    verifier_state
        .verifier_finalize_2(label, circuit, sub_claim_2, proof.eval_in_1)
        .expect("verifier phase2 finalization failed");
}
