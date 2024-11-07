use crate::{poly::dense::DenseMultilinearExtension, utils::math::Math};
use itertools::Itertools;
use rand::{prelude::SliceRandom, Rng};

use crate::{field::JoltField, utils::precompute_eq};

use super::sparse::MySparseMultilinearExtension;

/// Stores a multilinear polynomial in sparse evaluation form.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct TernarySparseMultilinearExtension<F: JoltField> {
    /// tuples of index and value
    pub evaluations_neg: Vec<usize>,
    pub evaluations_pos: Vec<usize>,
    /// number of variables
    pub num_vars: usize,
    zero: F,
}

impl<F: JoltField> TernarySparseMultilinearExtension<F> {
    pub fn from_evaluations(
        num_vars: usize,
        evaluations_neg: Vec<usize>,
        evaluations_pos: Vec<usize>,
    ) -> Self {
        Self {
            num_vars,
            evaluations_neg,
            evaluations_pos,
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
        only_positive: bool,
        rng: &mut R,
    ) -> Self {
        let size = 1 << num_vars;
        assert!(num_nonzero_entries <= size);

        // (0..num_nonzero_entries).for_each(|i| match rng.gen_range(0..2) {
        //     0 => evaluations_neg.push(rng.gen_range(0..size)),
        //     1 => evaluations_pos.push(rng.gen_range(0..size)),
        // });

        let mut indexes = (0..size).collect_vec();
        indexes.shuffle(rng);
        indexes.truncate(num_nonzero_entries);

        let (evaluations_neg, evaluations_pos) = if only_positive {
            (vec![], indexes)
        } else {
            // Half is -1, half is 1.
            let (left, right) = indexes.split_at(indexes.len() / 2);
            (left.to_vec(), right.to_vec())
        };

        Self {
            num_vars,
            evaluations_neg,
            evaluations_pos,
            zero: F::zero(),
        }
    }

    /// Convert the sparse multilinear polynomial to dense form.
    pub fn to_dense_multilinear_extension(&self) -> DenseMultilinearExtension<F> {
        let mut evaluations: Vec<_> = (0..(1 << self.num_vars)).map(|_| F::zero()).collect();
        for &i in self.evaluations_neg.iter() {
            evaluations[i] = -F::one();
        }
        for &i in self.evaluations_pos.iter() {
            evaluations[i] = F::one();
        }
        DenseMultilinearExtension::from_evaluations_vec(self.num_vars, evaluations)
    }
}

impl<F: JoltField> TernarySparseMultilinearExtension<F> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn window_size(&self) -> usize {
        ark_std::log2(self.evaluations_neg.len() + self.evaluations_pos.len()) as usize
    }

    fn evaluate(&self, point: &[F]) -> F {
        let keep_sparse_len = 0.max(self.num_vars - self.window_size());
        self.fix_variables(&point[..keep_sparse_len])
            .to_dense_multilinear_extension()
            .fix_variables(&point[keep_sparse_len..])[0]
    }

    /// Outputs an `l`-variate multilinear extension where value of evaluations
    /// are sampled uniformly at random. The number of nonzero entries is
    /// `sqrt(2^num_vars)` and indices of those nonzero entries are distributed
    /// uniformly at random.
    fn rand<R: Rng>(num_vars: usize, rng: &mut R) -> Self {
        Self::rand_with_config(num_vars, 1 << (num_vars / 2), false, rng)
    }

    pub fn fix_variables(&self, partial_point: &[F]) -> MySparseMultilinearExtension<F> {
        let dim = partial_point.len();
        assert!(dim <= self.num_vars, "invalid partial point dimension");

        let window =
            ark_std::log2(self.evaluations_neg.len() + self.evaluations_pos.len()) as usize;

        let focus_length = if partial_point.len() > window {
            window
        } else {
            partial_point.len()
        };
        let focus = &partial_point[..focus_length];
        let pre = precompute_eq(focus);

        let mut result = Vec::new();
        for src_entry in self.evaluations_neg.iter() {
            let old_idx = *src_entry;
            let gz = pre[old_idx & ((1 << focus_length) - 1)];
            let new_idx = old_idx >> focus_length;
            result.push((new_idx, -gz));
        }
        for src_entry in self.evaluations_pos.iter() {
            let old_idx = *src_entry;
            let gz = pre[old_idx & ((1 << focus_length) - 1)];
            let new_idx = old_idx >> focus_length;
            result.push((new_idx, gz));
        }

        let sparse = MySparseMultilinearExtension::from_evaluations_slice(
            self.num_vars - focus_length,
            &result,
        );
        if partial_point.len() == focus_length {
            sparse
        } else {
            sparse.fix_variables(&partial_point[focus_length..])
        }
    }

    pub fn to_evaluations(&self) -> Vec<F> {
        let mut evaluations: Vec<_> = (0..1 << self.num_vars).map(|_| F::zero()).collect();
        self.evaluations_neg
            .iter()
            .for_each(|&i| evaluations[i] = -F::one());
        self.evaluations_pos
            .iter()
            .for_each(|&i| evaluations[i] = F::one());
        evaluations
    }
}

#[cfg(test)]
mod test {
    use super::TernarySparseMultilinearExtension;
    type F = ark_bn254::Fr;

    #[test]
    fn test_fix_variables() {
        let num_vars = 3 * 2;
        let positive = vec![0, 8, 27];
        let negative = vec![60, 11, 13, 17, 19];
        let poly =
            TernarySparseMultilinearExtension::<F>::from_evaluations(num_vars, negative, positive);

        let point = vec![F::from(41), F::from(43), F::from(97)];
        let dense = poly.fix_variables(&point).to_dense_multilinear_extension();

        assert_eq!(
            dense.evaluations,
            vec![
                -F::from(161280 as u64),
                F::from(175002 as u64),
                F::from(3936 as u64),
                -F::from(169248 as u64),
                F::from(0 as u64),
                F::from(0 as u64),
                F::from(0 as u64),
                -F::from(162960 as u64),
            ]
        )
    }
}
