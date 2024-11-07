use std::ops::Range;

use ark_std::{end_timer, start_timer, test_rng};
use pcs::{
    bench_templates::{rand_ml_point, rand_ml_poly},
    commitment::{
        commitment_scheme::{BatchType, CommitShape, CommitmentScheme},
        hyrax::HyraxScheme,
    },
    utils::transcript::ProofTranscript,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const NUM_VARIABLES_RANGE: Range<usize> = 12..14;

type Fr = ark_bn254::Fr;

fn run<PCS: CommitmentScheme<Field = Fr>>(name: &'static str, num_vars: usize, sparsity: usize) {
    let rng = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();

    let r = rand_ml_point(2 * num_vars, rng);
    let mat_dense = rand_ml_poly(2 * num_vars, sparsity, rng);

    let mut prover_transcript = ProofTranscript::new(b"pcs");
    let srs = PCS::setup(&[CommitShape {
        input_length: 1 << (num_vars * 2),
        batch_type: BatchType::Small,
    }]);

    let timer =
        start_timer!(|| format!("{}, {}, 1/{}, normal commit", name, num_vars, 1 << sparsity));
    let comm = PCS::commit(&mat_dense, &srs);
    end_timer!(timer);

    let timer =
        start_timer!(|| format!("{}, {}, 1/{}, normal prove", name, num_vars, 1 << sparsity));
    let proof = PCS::prove(&srs, &mat_dense, &r, &mut prover_transcript);
    end_timer!(timer);

    let mut verifier_transcript = ProofTranscript::new(b"pcs");

    let mat_rx_ry = mat_dense.fix_variables(&r)[0];

    let timer =
        start_timer!(|| format!("{}, {}, 1/{}, normal verify", name, num_vars, 1 << sparsity));
    let _ = PCS::verify(
        &proof,
        &srs,
        &mut verifier_transcript,
        &r,
        &mat_rx_ry,
        &comm,
    );
    end_timer!(timer);
}

fn main() {
    for num_vars in NUM_VARIABLES_RANGE {
        for sparsity in (0..=4).rev() {
            run::<HyraxScheme<ark_bn254::G1Projective>>("hyrax", num_vars, sparsity);
        }
    }
}
