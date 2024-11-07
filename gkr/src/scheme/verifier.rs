use std::marker::PhantomData;

use ark_std::{end_timer, start_timer};
use pcs::{
    field::JoltField,
    utils::{precompute_eq, transcript::ProofTranscript},
};
use sumcheck::{
    error::Error,
    protocol::{verifier::SubClaim, PolynomialInfo},
    MLSumcheck, Proof,
};

use crate::circuit::Circuit;

pub struct VerifierState<F, C> {
    pub rz: Vec<F>,
    pub rx: Vec<F>,
    pub transcript: ProofTranscript,
    pub marker: PhantomData<C>,
}

impl<F: JoltField, C: Circuit<F>> VerifierState<F, C> {
    pub fn new(rz: Vec<F>) -> Self {
        Self {
            rz,
            rx: vec![],
            transcript: ProofTranscript::new(b"gkr"),
            marker: PhantomData,
        }
    }

    pub fn verifier_phase1(
        &mut self,
        label: (&'static str, usize, usize),
        identity_info: &PolynomialInfo,
        sum_1: F,
        proof_1: Proof<F>,
    ) -> Result<SubClaim<F>, Error> {
        let timer = start_timer!(|| format!(
            "{}, 1/{}: {} gkr phase1-verifier",
            label.1,
            1 << label.2,
            label.0
        ));
        let sub_claim = MLSumcheck::verify_as_subprotocol(
            &mut self.transcript,
            identity_info,
            sum_1,
            &proof_1,
        )?;
        end_timer!(timer);
        Ok(sub_claim)
    }

    pub fn verifier_finalize_1(
        &mut self,
        label: (&'static str, usize, usize),
        claim: SubClaim<F>,
        eval_in_0: F,
        eval_h: F,
    ) -> Result<(), Error> {
        let timer = start_timer!(|| format!(
            "{}, 1/{}: {} gkr final1-verifier",
            label.1,
            1 << label.2,
            label.0
        ));
        end_timer!(timer);
        self.rx = claim.point;
        self.transcript.append_scalar(&eval_in_0);
        self.transcript.append_scalar(&eval_h);
        if claim.expected_evaluation != eval_in_0 * eval_h {
            Err(Error::Reject(Some("phase 1".into())))
        } else {
            Ok(())
        }
    }

    pub fn verifier_phase2(
        &mut self,
        label: (&'static str, usize, usize),
        identity_info: &PolynomialInfo,
        sum_2: F,
        proof_2: Proof<F>,
    ) -> Result<SubClaim<F>, Error> {
        let timer = start_timer!(|| format!(
            "{}, 1/{}: {} gkr phase2-verifier",
            label.1,
            1 << label.2,
            label.0
        ));
        let claim = MLSumcheck::verify_as_subprotocol(
            &mut self.transcript,
            identity_info,
            sum_2,
            &proof_2,
        )?;
        end_timer!(timer);
        Ok(claim)
    }

    pub fn verifier_finalize_2(
        &mut self,
        label: (&'static str, usize, usize),
        circuit: &C,
        claim: SubClaim<F>,
        eval_in_1: F,
    ) -> Result<(), Error> {
        let timer = start_timer!(|| format!(
            "{}, 1/{}: {} gkr final2-verifier",
            label.1,
            1 << label.2,
            label.0
        ));
        let eq_rz = precompute_eq(&self.rz);
        let eq_rx = precompute_eq(&self.rx);
        let eq_ry = precompute_eq(&claim.point);
        let mul_gate_eval = circuit.compute_gate_eval(&eq_rz, &eq_rx, &eq_ry);
        self.transcript.append_scalar(&eval_in_1);
        end_timer!(timer);
        if claim.expected_evaluation != eval_in_1 * mul_gate_eval {
            Err(Error::Reject(Some("phase 2".into())))
        } else {
            Ok(())
        }
    }
}
