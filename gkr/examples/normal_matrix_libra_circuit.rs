use std::{marker::PhantomData, ops::Range, rc::Rc};

use ark_std::{test_rng, UniformRand, Zero};
use gkr::{
    circuit::{NormalCircuit, Witness},
    scheme::{gkr_prover, gkr_verifier},
};
use itertools::Itertools;
use pcs::poly::{dense::DenseMultilinearExtension, sparse::MySparseMultilinearExtension};

const NUM_VARIABLES_RANGE: Range<usize> = 12..14;

type F = ark_bn254::Fr;

fn run(num_vars: usize, sparsity: usize) {
    let mut rng = test_rng();
    let num_nonzero = 1 << (2 * num_vars - sparsity);

    // Generate random vector and matrix
    let vec: Vec<_> = (0..(1 << num_vars))
        .map(|_| F::rand(&mut rng))
        .collect_vec();
    let mat =
        MySparseMultilinearExtension::<F>::rand_with_config(2 * num_vars, num_nonzero, &mut rng);

    let mut res = vec![F::zero(); 1 << num_vars];
    mat.evaluations.iter().for_each(|&(pos, val)| {
        let x = pos >> num_vars;
        let y = pos & ((1 << num_vars) - 1);
        res[y] += vec[x] * val;
    });

    // Mul gates
    let size = 1 << num_vars;
    let gkr_num_vars = 2 * num_vars + 1;
    let mut mul_gate = vec![];
    for i in 0usize..size {
        for j in 0usize..size {
            mul_gate.push((j, i, size + ((i << num_vars) + j)));
        }
    }

    let mut mat_and_vec = [vec, mat.to_dense_multilinear_extension().evaluations].concat();
    mat_and_vec.extend(vec![F::zero(); (1 << gkr_num_vars) - mat_and_vec.len()]);

    let circuit = NormalCircuit {
        mul_gate,
        marker: PhantomData,
    };
    let circuit_witness = Witness {
        num_vars: gkr_num_vars,
        evaluations: Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            gkr_num_vars,
            mat_and_vec,
        )),
    };

    let rz = (0..num_vars).map(|_| F::rand(&mut rng)).collect_vec();
    let sum = DenseMultilinearExtension::from_evaluations_vec(num_vars, res).fix_variables(&rz)[0];
    let label = ("normal libra-circuit", num_vars, sparsity);

    let proof = gkr_prover(label, &circuit, circuit_witness, sum, &rz);
    gkr_verifier(label, &circuit, &rz, proof)
}

fn main() {
    for num_vars in NUM_VARIABLES_RANGE {
        for sparsity in (0..=4).rev() {
            run(num_vars, sparsity);
        }
    }
}
