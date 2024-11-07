use std::{ops::Range, rc::Rc};

use ark_std::{end_timer, start_timer, test_rng, One, UniformRand};
use itertools::{chain, Itertools};
use pcs::poly::{
    dense::DenseMultilinearExtension, sparse::MySparseMultilinearExtension,
    ternary_sparse::TernarySparseMultilinearExtension,
};
use sumcheck::{protocol::ListOfProductsOfPolynomials, MLSumcheck};

type F = ark_bn254::Fr;
const NUM_VARIABLES_RANGE: Range<usize> = 12..14;

fn run(num_vars: usize, sparsity: usize) {
    let num_nonzero = 1 << (2 * num_vars - sparsity);

    let rng = &mut test_rng();
    let ry = (0..num_vars).map(|_| F::rand(rng)).collect_vec();

    let mat_sparse = TernarySparseMultilinearExtension::<F>::rand_with_config(
        2 * num_vars,
        num_nonzero,
        false,
        rng,
    );
    let mat_dense = mat_sparse.clone().to_dense_multilinear_extension();
    let mat_normal = MySparseMultilinearExtension::from_evaluations_slice(
        2 * num_vars,
        &chain![
            mat_sparse.evaluations_neg.iter().map(|x| (*x, -F::one())),
            mat_sparse.evaluations_pos.iter().map(|x| (*x, F::one()))
        ]
        .collect_vec(),
    );

    let timer = start_timer!(|| format!(
        "{}, 1/{}: ternary matrix sumcheck dense setup",
        num_vars,
        1 << sparsity
    ));
    let mat_ry_1 = mat_dense.fix_variables(&ry);
    end_timer!(timer);

    let timer = start_timer!(|| format!(
        "{}, 1/{}: ternary matrix sumcheck ternary-sparse-setup",
        num_vars,
        1 << sparsity
    ));
    let mat_ry_2 = mat_sparse
        .fix_variables(&ry)
        .to_dense_multilinear_extension();
    end_timer!(timer);

    let timer = start_timer!(|| format!(
        "{}, 1/{}: ternary matrix sumcheck sparse-setup",
        num_vars,
        1 << sparsity
    ));
    let mat_ry_3 = mat_normal
        .fix_variables(&ry)
        .to_dense_multilinear_extension();
    end_timer!(timer);

    assert_eq!(mat_ry_1, mat_ry_2);
    assert_eq!(mat_ry_2, mat_ry_3);
    let mat_ry = Rc::new(mat_ry_1);
    let vec = Rc::new(DenseMultilinearExtension::rand(num_vars, rng));

    let timer = start_timer!(|| format!(
        "{}, 1/{}: ternary matrix sumcheck prover",
        num_vars,
        1 << sparsity,
    ));
    let mut identity = ListOfProductsOfPolynomials::new(num_vars);
    identity.add_product([vec.clone(), mat_ry], F::one());
    let proof = MLSumcheck::prove(&identity).expect("prover failed");
    end_timer!(timer);

    let sum = identity.expected_sum();
    let timer = start_timer!(|| format!(
        "{}, 1/{}: ternary matrix sumcheck verifier",
        num_vars,
        1 << sparsity,
    ));
    MLSumcheck::verify(&identity.info(), sum, &proof).expect("verifier failed");
    end_timer!(timer);
}

fn main() {
    for num_vars in NUM_VARIABLES_RANGE {
        for sparsity in (0..=4).rev() {
            run(num_vars, sparsity);
        }
    }
}
