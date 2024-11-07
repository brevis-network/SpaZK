use std::{marker::PhantomData, rc::Rc};

use itertools::chain;
use pcs::{field::JoltField, poly::dense::DenseMultilinearExtension};

pub trait Circuit<F: JoltField> {
    fn compute_phase1_h(&self, eq_rz: &[F], wit: &Witness<F>) -> Vec<F>;
    fn compute_phase2_h(&self, eq_rz: &[F], eq_rx: &[F], num_vars: usize) -> Vec<F>;
    fn compute_gate_eval(&self, eq_rz: &[F], eq_rx: &[F], eq_ry: &[F]) -> F;
}

pub struct NormalCircuit<F: JoltField> {
    pub mul_gate: Vec<(usize, usize, usize)>,
    pub marker: PhantomData<F>,
}

impl<F: JoltField> Circuit<F> for NormalCircuit<F> {
    /// Compute h(x) = sum_y mul(rz, x, y) * wit(y)
    fn compute_phase1_h(&self, eq_rz: &[F], wit: &Witness<F>) -> Vec<F> {
        let mut h = vec![F::zero(); 1 << wit.num_vars];
        for (o, i0, i1) in self.mul_gate.iter() {
            h[*i0] += eq_rz[*o] * wit.evaluations[*i1];
        }
        h
    }

    fn compute_phase2_h(&self, eq_rz: &[F], eq_rx: &[F], num_vars: usize) -> Vec<F> {
        let mut h = vec![F::zero(); 1 << num_vars];
        for (o, i0, i1) in self.mul_gate.iter() {
            h[*i1] += eq_rz[*o] * eq_rx[*i0];
        }
        h
    }

    fn compute_gate_eval(&self, eq_rz: &[F], eq_rx: &[F], eq_ry: &[F]) -> F {
        self.mul_gate
            .iter()
            .map(|(o, i0, i1)| eq_rz[*o] * eq_rx[*i0] * eq_ry[*i1])
            .sum::<F>()
    }
}

pub struct TernaryCircuit<F: JoltField> {
    pub mul_gate_pos: Vec<(usize, usize, usize)>,
    pub mul_gate_neg: Vec<(usize, usize, usize)>,
    pub marker: PhantomData<F>,
}

impl<F: JoltField> Circuit<F> for TernaryCircuit<F> {
    /// Compute h(x) = sum_y mul(rz, x, y) * wit(y)
    fn compute_phase1_h(&self, eq_rz: &[F], wit: &Witness<F>) -> Vec<F> {
        let mut h = vec![F::zero(); 1 << wit.num_vars];
        for (o, i0, _) in self.mul_gate_pos.iter() {
            h[*i0] += eq_rz[*o];
        }
        for (o, i0, _) in self.mul_gate_neg.iter() {
            h[*i0] -= eq_rz[*o];
        }
        h
    }

    fn compute_phase2_h(&self, eq_rz: &[F], eq_rx: &[F], num_vars: usize) -> Vec<F> {
        let mut h = vec![F::zero(); 1 << num_vars];
        for (o, i0, i1) in chain![&self.mul_gate_pos, &self.mul_gate_neg] {
            h[*i1] += eq_rz[*o] * eq_rx[*i0]
        }
        h
    }

    fn compute_gate_eval(&self, eq_rz: &[F], eq_rx: &[F], eq_ry: &[F]) -> F {
        chain![&self.mul_gate_pos, &self.mul_gate_neg]
            .map(|(o, i0, i1)| eq_rz[*o] * eq_rx[*i0] * eq_ry[*i1])
            .sum::<F>()
    }
}

pub struct Witness<F: JoltField> {
    pub num_vars: usize,
    pub evaluations: Rc<DenseMultilinearExtension<F>>,
}
