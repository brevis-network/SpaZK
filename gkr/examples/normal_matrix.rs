use std::{ops::Range, rc::Rc};

use ark_std::{end_timer, start_timer, test_rng, One, UniformRand};
use itertools::Itertools;
use pcs::{
    poly::{dense::DenseMultilinearExtension, sparse::MySparseMultilinearExtension},
    utils::transcript::ProofTranscript,
};
use sumcheck::{protocol::ListOfProductsOfPolynomials, MLSumcheck};

const NUM_VARIABLES_RANGE: Range<usize> = 12..14;

type Fr = ark_bn254::Fr;

fn run(num_vars: usize, sparsity: usize) {
    let num_nonzero = 1 << (2 * num_vars - sparsity);
    let rng = &mut test_rng();
    let vec = Rc::new(DenseMultilinearExtension::<Fr>::rand(num_vars, rng));
    let ry = (0..num_vars).map(|_| Fr::rand(rng)).collect_vec();

    // Sparse setup and dense setup
    let mat_sparse =
        MySparseMultilinearExtension::<Fr>::rand_with_config(2 * num_vars, num_nonzero, rng);
    let mat_dense = mat_sparse.clone().to_dense_multilinear_extension();

    let timer = start_timer!(|| format!(
        "{}, 1/{}: normal matrix sumcheck dense-setup",
        num_vars,
        1 << sparsity
    ));
    let _ = Rc::new(mat_dense.fix_variables(&ry));
    end_timer!(timer);

    let timer = start_timer!(|| format!(
        "{}, 1/{}: normal matrix sumcheck sparse-setup",
        num_vars,
        1 << sparsity
    ));
    let mat_ry = Rc::new(
        mat_sparse
            .fix_variables(&ry)
            .to_dense_multilinear_extension(),
    );
    end_timer!(timer);

    let mut prover_transcript = ProofTranscript::new(b"test");

    let timer = start_timer!(|| format!(
        "{}, 1/{}: normal matrix sumcheck prover",
        num_vars,
        1 << sparsity
    ));
    let mut identity = ListOfProductsOfPolynomials::new(num_vars);
    identity.add_product([vec, mat_ry], Fr::one());
    let (proof, state) =
        MLSumcheck::prove_as_subprotocol(&mut prover_transcript, &identity).expect("prover failed");
    end_timer!(timer);

    let vec_rx =
        state.flattened_ml_extensions[0].fix_variables(&[*state.randomness.last().unwrap()])[0];

    let sum = identity.expected_sum();
    let mut verifier_transcript = ProofTranscript::new(b"test");

    let timer = start_timer!(|| format!(
        "{}, 1/{}: normal matrix sumcheck verifier",
        num_vars,
        1 << sparsity
    ));
    let claim =
        MLSumcheck::verify_as_subprotocol(&mut verifier_transcript, &identity.info(), sum, &proof)
            .expect("verifier failed");
    end_timer!(timer);

    let timer = start_timer!(|| format!(
        "{}, 1/{}: normal matrix sumcheck dense finalization",
        num_vars,
        1 << sparsity
    ));
    // TODO: add polynomial commitment.
    let mat_rx_ry = mat_sparse
        .fix_variables(&ry)
        .to_dense_multilinear_extension()
        .fix_variables(&claim.point)[0];
    assert_eq!(claim.expected_evaluation, mat_rx_ry * vec_rx);
    end_timer!(timer);
}

fn main() {
    for num_vars in NUM_VARIABLES_RANGE {
        for sparsity in (0..=4).rev() {
            run(num_vars, sparsity);
        }
    }
}
