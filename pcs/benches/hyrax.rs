use ark_bn254::G1Projective;
use criterion::{criterion_group, criterion_main, Criterion};
use pcs::{bench, bench_method, bench_templates::*, commitment::hyrax::HyraxScheme};

const MIN_NUM_VARS: usize = 24;
const MAX_NUM_VARS: usize = 26;

const MIN_SPARSITY: usize = 0;
const MAX_SPARSITY: usize = 4;

bench!(
    HyraxScheme<G1Projective>,
    rand_ternary_ml_poly,
    rand_ml_point
);
