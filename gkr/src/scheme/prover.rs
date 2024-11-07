use std::{marker::PhantomData, rc::Rc};

use ark_std::{end_timer, start_timer};
use pcs::{
    field::JoltField,
    poly::dense::DenseMultilinearExtension,
    utils::{precompute_eq, transcript::ProofTranscript},
};
use sumcheck::{error::Error, protocol::ListOfProductsOfPolynomials, MLSumcheck, Proof};

use crate::circuit::{Circuit, Witness};
pub struct ProverState<F, C> {
    pub eq_rz: Vec<F>,
    pub transcript: ProofTranscript,
    pub marker: PhantomData<C>,
}

impl<F: JoltField, C: Circuit<F>> Default for ProverState<F, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: JoltField, C: Circuit<F>> ProverState<F, C> {
    pub fn new() -> Self {
        Self {
            eq_rz: vec![],
            transcript: ProofTranscript::new(b"gkr"),
            marker: PhantomData,
        }
    }

    /// phase 1: out(rz) = \sum_x in(x) * \sum_y mul(rz, x, y) * in(y)
    pub fn prover_phase1_setup(
        &mut self,
        label: (&'static str, usize, usize),
        circuit: &C,
        wit: &Witness<F>,
        rz: &[F],
    ) -> ListOfProductsOfPolynomials<F> {
        let timer = start_timer!(|| format!(
            "{}, 1/{}: {} gkr phase1-setup",
            label.1,
            1 << label.2,
            label.0
        ));
        let eq_rz = precompute_eq(rz);

        let polys = [
            wit.evaluations.clone(),
            Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                wit.num_vars,
                circuit.compute_phase1_h(&eq_rz, wit),
            )),
        ];
        self.eq_rz = eq_rz;
        let mut identity = ListOfProductsOfPolynomials::new(wit.num_vars);
        identity.add_product(polys, F::one());
        end_timer!(timer);
        identity
    }

    pub fn prover_phase1(
        &mut self,
        label: (&'static str, usize, usize),
        identity: &ListOfProductsOfPolynomials<F>,
    ) -> Result<(Proof<F>, Vec<F>, F, F), Error> {
        let timer = start_timer!(|| format!(
            "{}, 1/{}: {} gkr phase1-prover",
            label.1,
            1 << label.2,
            label.0
        ));
        let (sc_proof, sc_state) =
            MLSumcheck::prove_as_subprotocol(&mut self.transcript, identity)?;
        let eval_in_0 = sc_state.final_evaluation(0);
        let eval_h = sc_state.final_evaluation(1);

        self.transcript.append_scalar(&eval_in_0);
        self.transcript.append_scalar(&eval_h);

        end_timer!(timer);
        Ok((sc_proof, sc_state.randomness, eval_in_0, eval_h))
    }

    /// phase 2: polys[1](rx) = \sum_y mul(rz, rx, y) * in(y)
    pub fn prover_phase2_setup(
        &mut self,
        label: (&'static str, usize, usize),
        circuit: &C,
        wit: &Witness<F>,
        rx: &[F],
    ) -> ListOfProductsOfPolynomials<F> {
        let timer = start_timer!(|| format!(
            "{}, 1/{}: {} gkr phase2-setup",
            label.1,
            1 << label.2,
            label.0
        ));
        let eq_rx = precompute_eq(rx);
        let polys = [
            wit.evaluations.clone(),
            Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                wit.num_vars,
                circuit.compute_phase2_h(&self.eq_rz, &eq_rx, wit.num_vars),
            )),
        ];
        let mut identity = ListOfProductsOfPolynomials::new(wit.num_vars);
        identity.add_product(polys, F::one());
        end_timer!(timer);
        identity
    }

    pub fn prover_phase2(
        &mut self,
        label: (&'static str, usize, usize),
        identity: &ListOfProductsOfPolynomials<F>,
    ) -> Result<(Proof<F>, Vec<F>, F), Error> {
        let timer = start_timer!(|| format!(
            "{}, 1/{}: {} gkr phase2-prover",
            label.1,
            1 << label.2,
            label.0
        ));
        let (sc_proof, sc_state) =
            MLSumcheck::prove_as_subprotocol(&mut self.transcript, identity)?;
        let eval_in_1 = sc_state.final_evaluation(0);

        self.transcript.append_scalar(&eval_in_1);

        end_timer!(timer);
        Ok((sc_proof, sc_state.randomness, eval_in_1))
    }
}
