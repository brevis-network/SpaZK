use std::time::{Duration, Instant};

use crate::{
    commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme},
    field::JoltField,
    poly::{
        dense::DenseMultilinearExtension, sparse::MySparseMultilinearExtension,
        ternary_sparse::TernarySparseMultilinearExtension,
    },
    utils::transcript::ProofTranscript,
};
use ark_std::test_rng;
use criterion::{BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

pub fn rand_ml_point<F: JoltField>(num_vars: usize, rng: &mut ChaCha20Rng) -> Vec<F> {
    (0..num_vars).map(|_| F::random(rng)).collect()
}

pub fn rand_ml_poly<F: JoltField>(
    num_vars: usize,
    sparsity: usize,
    rng: &mut ChaCha20Rng,
) -> DenseMultilinearExtension<F> {
    let num_nonzero_entries = 1 << (num_vars - sparsity);
    MySparseMultilinearExtension::rand_with_config(num_vars, num_nonzero_entries, rng)
        .to_dense_multilinear_extension()
}

pub fn rand_ternary_ml_poly<F: JoltField>(
    num_vars: usize,
    sparsity: usize,
    rng: &mut ChaCha20Rng,
) -> DenseMultilinearExtension<F> {
    let num_nonzero_entries = 1 << (num_vars - sparsity);
    TernarySparseMultilinearExtension::rand_with_config(num_vars, num_nonzero_entries, false, rng)
        .to_dense_multilinear_extension()
}

#[derive(Clone, Copy, Debug)]
struct Pars {
    num_vars: usize,
    sparsity: usize,
}

impl std::fmt::Display for Pars {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "nv:{}/sp:{}", self.num_vars, self.sparsity)
    }
}

pub fn bench_pcs_method<PCS: CommitmentScheme>(
    c: &mut Criterion,
    nv_list: Vec<usize>,
    sp_list: Vec<usize>,
    msg: &str,
    method: impl Fn(
        &PCS::Setup,
        usize,
        usize,
        fn(usize, usize, &mut ChaCha20Rng) -> DenseMultilinearExtension<PCS::Field>,
        fn(usize, &mut ChaCha20Rng) -> Vec<PCS::Field>,
    ) -> Duration,
    rand_poly: fn(usize, usize, &mut ChaCha20Rng) -> DenseMultilinearExtension<PCS::Field>,
    rand_point: fn(usize, &mut ChaCha20Rng) -> Vec<PCS::Field>,
) {
    let mut group = c.benchmark_group(msg);
    group.sample_size(10);
    for num_vars in nv_list {
        for &sparsity in sp_list.iter() {
            let pp = PCS::setup(&[CommitShape::new(1 << num_vars, BatchType::Small)]);
            group.bench_with_input(
                BenchmarkId::from_parameter(Pars { num_vars, sparsity }),
                &(num_vars, sparsity),
                |b, (num_vars, sparsity)| {
                    b.iter_custom(|i| {
                        let mut time = Duration::from_nanos(0);
                        for _ in 0..i {
                            time += method(&pp, *num_vars, *sparsity, rand_poly, rand_point);
                        }
                        time
                    })
                },
            );
        }
    }
    group.finish();
}

pub fn commit<PCS: CommitmentScheme>(
    pp: &PCS::Setup,
    num_vars: usize,
    sparsity: usize,
    rand_poly: fn(usize, usize, &mut ChaCha20Rng) -> DenseMultilinearExtension<PCS::Field>,
    _rand_point: fn(usize, &mut ChaCha20Rng) -> Vec<PCS::Field>,
) -> Duration {
    let rng = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();
    let poly = rand_poly(num_vars, sparsity, rng);
    let start = Instant::now();
    let _ = PCS::commit(&poly, pp);
    start.elapsed()
}

pub fn open<PCS: CommitmentScheme>(
    pp: &PCS::Setup,
    num_vars: usize,
    sparsity: usize,
    rand_poly: fn(usize, usize, &mut ChaCha20Rng) -> DenseMultilinearExtension<PCS::Field>,
    rand_point: fn(usize, &mut ChaCha20Rng) -> Vec<PCS::Field>,
) -> Duration {
    let rng = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();
    let poly = rand_poly(num_vars, sparsity, rng);
    let _ = PCS::commit(&poly, pp);

    let point = rand_point(num_vars, rng);
    let mut transcript = ProofTranscript::new(b"pcs");
    let start = Instant::now();
    let _ = PCS::prove(pp, &poly, &point, &mut transcript);
    start.elapsed()
}

pub fn verify<PCS: CommitmentScheme>(
    pp: &PCS::Setup,
    num_vars: usize,
    sparsity: usize,
    rand_poly: fn(usize, usize, &mut ChaCha20Rng) -> DenseMultilinearExtension<PCS::Field>,
    rand_point: fn(usize, &mut ChaCha20Rng) -> Vec<PCS::Field>,
) -> Duration {
    let rng = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();
    let poly = rand_poly(num_vars, sparsity, rng);
    let comm = PCS::commit(&poly, pp);

    let point = rand_point(num_vars, rng);
    let mut transcript = ProofTranscript::new(b"pcs");
    let proof = PCS::prove(pp, &poly, &point, &mut transcript);
    let eval = poly.fix_variables(&point)[0];
    let start = Instant::now();
    let mut transcript = ProofTranscript::new(b"pcs");
    let _ = PCS::verify(&proof, pp, &mut transcript, &point, &eval, &comm);
    start.elapsed()
}

#[macro_export]
macro_rules! bench_method {
    ($c:expr, $method:ident, $scheme_type:ty, $rand_poly:ident, $rand_point:ident) => {
        let scheme_type_str = stringify!($scheme_type);
        let bench_name = format!("{}/{}", stringify!($method), scheme_type_str);
        bench_pcs_method::<$scheme_type>(
            $c,
            (MIN_NUM_VARS..=MAX_NUM_VARS).step_by(2).collect(),
            (MIN_SPARSITY..=MAX_SPARSITY).rev().collect(),
            &bench_name,
            $method::<$scheme_type>,
            $rand_poly,
            $rand_point,
        )
    };
}

#[macro_export]
macro_rules! bench {
    ($scheme_type:ty, $rand_poly:ident, $rand_point:ident) => {
        fn bench_pcs(c: &mut Criterion) {
            bench_method!(c, commit, $scheme_type, $rand_poly, $rand_point);
            bench_method!(c, open, $scheme_type, $rand_poly, $rand_point);
            bench_method!(c, verify, $scheme_type, $rand_poly, $rand_point);
        }

        criterion_group!(benches, bench_pcs);
        criterion_main!(benches);
    };
}
