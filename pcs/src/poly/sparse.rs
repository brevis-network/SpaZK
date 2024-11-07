use itertools::Itertools;
use rand::{prelude::SliceRandom, Rng};

use crate::{field::JoltField, utils::precompute_eq};

use super::dense::DenseMultilinearExtension;

/// Stores a multilinear polynomial in sparse evaluation form.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct MySparseMultilinearExtension<F: JoltField> {
    /// tuples of index and value
    pub evaluations: Vec<(usize, F)>,
    /// number of variables
    pub num_vars: usize,
    zero: F,
}

impl<F: JoltField> MySparseMultilinearExtension<F> {
    pub fn from_evaluations_slice<'a>(
        num_vars: usize,
        evaluations: impl IntoIterator<Item = &'a (usize, F)>,
    ) -> Self {
        let bit_mask = 1 << num_vars;
        let evaluations = evaluations.into_iter();
        let evaluations: Vec<_> = evaluations
            .map(|(i, v): &(usize, F)| {
                assert!(*i < bit_mask, "index out of range");
                (*i, *v)
            })
            .collect();

        Self {
            evaluations,
            num_vars,
            zero: F::zero(),
        }
    }

    pub fn from_evaluations_vec(num_vars: usize, evaluations: Vec<(usize, F)>) -> Self {
        let bit_mask = 1 << num_vars;
        evaluations.iter().for_each(|(i, _)| {
            assert!(*i < bit_mask, "index out of range");
        });

        Self {
            evaluations,
            num_vars,
            zero: F::zero(),
        }
    }

    /// Outputs an `l`-variate multilinear extension where value of evaluations
    /// are sampled uniformly at random. The number of nonzero entries is
    /// `num_nonzero_entries` and indices of those nonzero entries are
    /// distributed uniformly at random.
    ///
    /// Note that this function uses rejection sampling. As number of nonzero
    /// entries approach `2 ^ num_vars`, sampling will be very slow due to
    /// large number of collisions.
    pub fn rand_with_config<R: Rng>(
        num_vars: usize,
        num_nonzero_entries: usize,
        rng: &mut R,
    ) -> Self {
        assert!(num_nonzero_entries <= (1 << num_vars));

        let mut range: Vec<usize> = (0..(1 << num_vars)).collect_vec();
        range.shuffle(rng);

        let evaluations = (0..num_nonzero_entries)
            .map(|i| (range[i], F::random(rng)))
            .collect_vec();
        Self {
            num_vars,
            evaluations,
            zero: F::zero(),
        }
    }

    /// Convert the sparse multilinear polynomial to dense form.
    pub fn to_dense_multilinear_extension(&self) -> DenseMultilinearExtension<F> {
        let mut evaluations: Vec<_> = (0..(1 << self.num_vars)).map(|_| F::zero()).collect();
        for (i, v) in self.evaluations.iter() {
            evaluations[*i] += *v;
        }
        DenseMultilinearExtension::from_evaluations_vec(self.num_vars, evaluations)
    }
}

impl<F: JoltField> MySparseMultilinearExtension<F> {
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn window_size(&self) -> usize {
        ark_std::log2(self.evaluations.len()) as usize
    }

    pub fn evaluate(&self, point: &[F]) -> F {
        let keep_sparse_len = 0.max(self.num_vars - self.window_size());
        self.fix_variables(&point[..keep_sparse_len])
            .to_dense_multilinear_extension()
            .fix_variables(&point[keep_sparse_len..])[0]
    }

    /// Outputs an `l`-variate multilinear extension where value of evaluations
    /// are sampled uniformly at random. The number of nonzero entries is
    /// `sqrt(2^num_vars)` and indices of those nonzero entries are distributed
    /// uniformly at random.
    pub fn rand<R: Rng>(num_vars: usize, rng: &mut R) -> Self {
        Self::rand_with_config(num_vars, 1 << (num_vars / 2), rng)
    }

    pub fn fix_variables(&self, partial_point: &[F]) -> Self {
        let dim = partial_point.len();
        assert!(dim <= self.num_vars, "invalid partial point dimension");

        let window = self.window_size();
        let mut point = partial_point;
        let mut last = self.evaluations.clone();
        // batch evaluation
        while !point.is_empty() {
            let focus_length = if point.len() > window {
                window
            } else {
                point.len()
            };
            let focus = &point[..focus_length];
            point = &point[focus_length..];
            let pre = precompute_eq(focus);
            let mut result = Vec::new();
            for src_entry in last.iter() {
                let old_idx = src_entry.0;
                let gz = pre[old_idx & ((1 << focus_length) - 1)];
                let new_idx = old_idx >> focus_length;
                result.push((new_idx, gz * src_entry.1));
            }
            last = result;
        }
        Self {
            num_vars: self.num_vars - partial_point.len(),
            evaluations: last,
            zero: F::zero(),
        }
    }

    pub fn to_evaluations(&self) -> Vec<F> {
        let mut evaluations: Vec<_> = (0..1 << self.num_vars).map(|_| F::zero()).collect();
        for (i, v) in self.evaluations.iter() {
            evaluations[*i] = *v;
        }
        evaluations
    }
}

#[cfg(test)]
mod test {
    use super::MySparseMultilinearExtension;

    type F = ark_bn254::Fr;
    #[test]
    fn test_fix_variables_to() {
        let num_vars = 3 * 2;
        let evaluations = vec![(0usize, F::from(77)), (8, -F::from(31)), (47, -F::from(97))];
        let poly =
            MySparseMultilinearExtension::<F>::from_evaluations_slice(num_vars, &evaluations);

        let point = vec![F::from(41), F::from(43), F::from(97)];
        let dense = poly.fix_variables(&point).to_dense_multilinear_extension();

        assert_eq!(dense.num_vars, 3);
        assert_eq!(
            dense.evaluations,
            vec![
                -F::from(12418560 as u64),
                F::from(4999680 as u64),
                F::from(0 as u64),
                F::from(0 as u64),
                F::from(0 as u64),
                -F::from(16588067 as u64),
                F::from(0 as u64),
                F::from(0 as u64),
            ]
        )
    }
}
