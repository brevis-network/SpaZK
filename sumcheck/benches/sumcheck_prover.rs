use std::{ops::Range, rc::Rc};

use ark_bn254::Fr;
use ark_std::{test_rng, One};
use criterion::{criterion_group, criterion_main, Criterion};
use pcs::{
    field::JoltField,
    poly::{dense::DenseMultilinearExtension, eq_poly::EqPolynomial},
};
use sumcheck::{protocol::ListOfProductsOfPolynomials, MLSumcheck};

const NV_RANGE: Range<usize> = 19..21;

fn benchmark_sumcheck(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut group = c.benchmark_group("sumcheck");
    group.sample_size(10);
    for nv in NV_RANGE {
        let poly = Rc::new(DenseMultilinearExtension::rand(nv, &mut rng));
        let mut identity = ListOfProductsOfPolynomials::new(nv);
        identity.add_product([poly], Fr::one());
        group.bench_function(format!("{}", nv).as_str(), |b| {
            b.iter(|| MLSumcheck::prove(&identity));
        });
    }
}

fn benchmark_zerocheck(c: &mut Criterion) {
    let mut rng = test_rng();
    let mut group = c.benchmark_group("zerocheck");
    group.sample_size(10);
    for nv in NV_RANGE {
        let poly_a = Rc::new(DenseMultilinearExtension::rand(nv, &mut rng));
        let poly_b = Rc::new(DenseMultilinearExtension::rand(nv, &mut rng));
        let point = (0..nv).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
        let eq = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            nv,
            EqPolynomial::evals(&point),
        ));
        let mut identity = ListOfProductsOfPolynomials::new(nv);
        identity.add_product([eq, poly_a, poly_b], Fr::one());
        group.bench_function(format!("{}", nv).as_str(), |b| {
            b.iter(|| MLSumcheck::prove(&identity));
        });
    }
}

criterion_group!(benches, benchmark_sumcheck, benchmark_zerocheck);
criterion_main!(benches);
