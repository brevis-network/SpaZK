use std::{fmt::Debug, iter::Sum, ops::*};

use ark_std::Zero;
use itertools::{izip, Itertools};
use num_traits::Signed;
use pcs::{
    field::JoltField,
    poly::{
        dense::DenseMultilinearExtension, sparse::MySparseMultilinearExtension,
        ternary_sparse::TernarySparseMultilinearExtension,
    },
    utils::math::Math,
};
use rand::{Rng, RngCore};

pub trait TensorData:
    Clone
    + Debug
    + Copy
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + Zero
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + Sum
    + PartialEq
    + Signed
    + From<bool>
{
    const BIT_WIDTH: usize;
    const PRECISION: usize;
    const RANGE_BIT_WIDTH: usize;

    fn random<R: RngCore>(rng: &mut R) -> Self;
    fn to_range_vector(&self) -> Vec<Self>;
    fn scaled_by(&self, n_bit: usize) -> Self;
    fn divided_by(&self, n_bit: usize) -> Self;
    fn max(&self, other: &Self) -> Self;
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tensors<T>(pub Vec<Tensor<T>>);

pub struct TensorClaim<F: JoltField> {
    pub lo_point: Vec<F>,
    pub hi_point: Vec<F>,
    pub value: F,
}

#[derive(Clone, Debug, Default)]
pub struct SparseTensor<T> {
    pub data: Vec<(usize, T)>,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct TernarySparseTensor {
    pub pos: Vec<usize>,
    pub neg: Vec<usize>,
    pub shape: Vec<usize>,
}

impl<T: Clone + Copy> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        Tensor { data, shape }
    }

    pub fn num_vars(&self) -> usize {
        self.data.len().log_2()
    }

    pub fn scaled_by(&self, n_bit: usize) -> Self
    where
        T: TensorData,
    {
        let data = self.data.iter().map(|d| d.scaled_by(n_bit)).collect_vec();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn divided_by(&self, n_bit: usize) -> Self
    where
        T: TensorData,
    {
        let data = self.data.iter().map(|d| d.divided_by(n_bit)).collect_vec();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn index(&self, index: &[usize]) -> T {
        self.data[izip!(index.iter(), self.shape.iter()).fold(0, |acc, (i, s)| acc * s + i)]
    }

    pub fn is_positive(&self) -> Tensor<T>
    where
        T: TensorData,
    {
        Tensor {
            data: self
                .data
                .iter()
                .map(|x| x.is_positive().into())
                .collect_vec(),
            shape: self.shape.clone(),
        }
    }

    pub fn relu(&self) -> Tensor<T>
    where
        T: TensorData,
    {
        Tensor {
            data: self.data.iter().map(|x| (*x).max(&T::zero())).collect_vec(),
            shape: self.shape.clone(),
        }
    }

    pub fn relu_and_rescale(&self, precision: usize) -> Tensor<T>
    where
        T: TensorData,
    {
        Tensor {
            data: self
                .data
                .iter()
                .map(|x| (*x).max(&T::zero()).divided_by(precision))
                .collect_vec(),
            shape: self.shape.clone(),
        }
    }

    pub fn to_range_vectors(&self) -> Vec<Tensor<T>>
    where
        T: TensorData,
    {
        assert!(!self.data.is_empty());
        let mut range_vectors = self.data[0]
            .to_range_vector()
            .into_iter()
            .map(|x| vec![x.abs()])
            .collect_vec();
        for item in self.data.iter().skip(1) {
            let new_range_vector = item.abs().to_range_vector();
            for (prev, curr) in izip!(range_vectors.iter_mut(), new_range_vector.into_iter()) {
                prev.push(curr);
            }
        }

        range_vectors
            .into_iter()
            .map(|mut data| {
                let len = data.len().next_power_of_two();
                data.extend(vec![T::zero(); len - data.len()]);
                Tensor {
                    data,
                    shape: vec![1, len],
                }
            })
            .collect_vec()
    }
}

impl<T: Clone + Copy> Tensors<T> {
    pub fn new(tensors: Vec<Tensor<T>>) -> Self {
        // all tensor has the same shape
        if !tensors.is_empty() {
            assert!(tensors
                .iter()
                .all(|tensor| tensor.num_vars() == tensors[0].num_vars()));
        }
        Tensors(tensors)
    }

    pub fn num_vars(&self) -> usize {
        assert!(!self.0.is_empty());
        self.lo_num_vars() + self.hi_num_vars()
    }

    /// Index the cell within an instance.
    pub fn hi_num_vars(&self) -> usize {
        assert!(!self.0.is_empty());
        self.0[0].num_vars()
    }

    /// Index the instances.
    pub fn lo_num_vars(&self) -> usize {
        assert!(!self.0.is_empty());
        self.0.len().log_2()
    }

    pub fn scaled_by(&self, n_bit: usize) -> Self
    where
        T: TensorData,
    {
        let tensors = self
            .0
            .iter()
            .map(|tensor| Tensor {
                data: tensor.data.iter().map(|d| d.scaled_by(n_bit)).collect_vec(),
                shape: tensor.shape.clone(),
            })
            .collect_vec();
        Tensors(tensors)
    }

    pub fn is_positive(&self) -> Tensors<T>
    where
        T: TensorData,
    {
        Tensors(
            ark_std::cfg_iter!(self.0)
                .map(|tensor| tensor.is_positive())
                .collect(),
        )
    }

    pub fn relu(&self) -> Tensors<T>
    where
        T: TensorData,
    {
        Tensors(
            ark_std::cfg_iter!(self.0)
                .map(|tensor: &Tensor<T>| tensor.relu())
                .collect(),
        )
    }

    pub fn relu_and_rescale(&self, precision: usize) -> Tensors<T>
    where
        T: TensorData,
    {
        Tensors(
            ark_std::cfg_iter!(self.0)
                .map(|tensor: &Tensor<T>| tensor.relu_and_rescale(precision))
                .collect(),
        )
    }

    pub fn to_range_vectors(&self) -> Vec<Tensors<T>>
    where
        T: TensorData,
    {
        let mut range_vectors = self.0[0]
            .to_range_vectors()
            .into_iter()
            .map(|item| vec![item])
            .collect_vec();

        for item in self.0.iter().skip(1) {
            let new_range_vectors = item.to_range_vectors();
            for (prev, curr) in izip!(range_vectors.iter_mut(), new_range_vectors) {
                prev.push(curr);
            }
        }

        range_vectors
            .into_iter()
            .map(|item| Tensors(item))
            .collect_vec()
    }
}

impl<T> SparseTensor<T> {
    pub fn new(data: Vec<(usize, T)>, shape: Vec<usize>) -> Self {
        SparseTensor { data, shape }
    }

    pub fn num_vars(&self) -> usize {
        self.shape.iter().product::<usize>().log_2()
    }
}

impl TernarySparseTensor {
    pub fn new(pos: Vec<usize>, neg: Vec<usize>, shape: Vec<usize>) -> Self {
        TernarySparseTensor { pos, neg, shape }
    }

    pub fn num_vars(&self) -> usize {
        self.shape.iter().product::<usize>().log_2()
    }
}

impl<T: TensorData> Add<Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, other: Tensor<T>) -> Tensor<T> {
        add(&self, &other)
    }
}

impl<'a, T: TensorData> Add<Tensor<T>> for &'a Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, other: Tensor<T>) -> Tensor<T> {
        add(self, &other)
    }
}

impl<'b, T: TensorData> Add<&'b Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, other: &'b Tensor<T>) -> Tensor<T> {
        add(&self, other)
    }
}

impl<'a, 'b, T: TensorData> Add<&'b Tensor<T>> for &'a Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, other: &'b Tensor<T>) -> Tensor<T> {
        add(self, other)
    }
}

impl<T: TensorData> Mul<Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, other: Tensor<T>) -> Tensor<T> {
        mul(&self, &other)
    }
}

impl<'a, T: TensorData> Mul<Tensor<T>> for &'a Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, other: Tensor<T>) -> Tensor<T> {
        mul(self, &other)
    }
}

impl<'b, T: TensorData> Mul<&'b Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, other: &'b Tensor<T>) -> Tensor<T> {
        mul(&self, other)
    }
}

impl<'a, 'b, T: TensorData> Mul<&'b Tensor<T>> for &'a Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, other: &'b Tensor<T>) -> Tensor<T> {
        mul(self, other)
    }
}

impl<'b, T: TensorData> Mul<&'b SparseTensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, other: &'b SparseTensor<T>) -> Tensor<T> {
        mul_sparse(&self, other)
    }
}

impl<'b, T: TensorData> Mul<&'b TernarySparseTensor> for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, other: &'b TernarySparseTensor) -> Tensor<T> {
        mul_ternary_sparse(&self, other)
    }
}

impl<'a, 'b, T: TensorData> Mul<&'b SparseTensor<T>> for &'a Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, other: &'b SparseTensor<T>) -> Tensor<T> {
        mul_sparse(self, other)
    }
}

impl<'a, 'b, T: TensorData> Mul<&'b TernarySparseTensor> for &'a Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, other: &'b TernarySparseTensor) -> Tensor<T> {
        mul_ternary_sparse(self, other)
    }
}

fn add<T: TensorData>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    assert_eq!(a.shape, b.shape);

    let data = izip!(ark_std::cfg_iter!(a.data), ark_std::cfg_iter!(b.data))
        .map(|(x, y)| *x + y)
        .collect_vec();
    Tensor {
        data,
        shape: a.shape.clone(),
    }
}

fn mul<T: TensorData>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    assert_eq!(a.shape.len(), 2);
    assert_eq!(b.shape.len(), 2);
    assert_eq!(a.shape[1], b.shape[0]);

    let data = ark_std::cfg_into_iter!(0..a.shape[0])
        .flat_map(|i| {
            ark_std::cfg_into_iter!(0..b.shape[1])
                .map(|k| {
                    (0..a.shape[1])
                        .map(|j| a.index(&[i, j]) * b.index(&[j, k]))
                        .sum::<T>()
                })
                .collect_vec()
        })
        .collect_vec();

    Tensor {
        data,
        shape: vec![a.shape[0], b.shape[1]],
    }
}

fn mul_sparse<T: TensorData>(a: &Tensor<T>, b: &SparseTensor<T>) -> Tensor<T> {
    assert_eq!(a.shape.len(), 2);
    assert_eq!(b.shape.len(), 2);
    assert_eq!(a.shape[1], b.shape[0]);

    let mut data = vec![T::zero(); a.shape[0] * b.shape[1]];
    ark_std::cfg_into_iter!(0..b.data.len()).for_each(|i| {
        let (idx, value) = &b.data[i];
        let x = idx / b.shape[1];
        let y = idx % b.shape[1];
        ark_std::cfg_into_iter!(0..a.shape[0])
            .for_each(|k| data[k * b.shape[1] + y] += a.index(&[k, x]) * *value);
    });

    Tensor {
        data,
        shape: vec![a.shape[0], b.shape[1]],
    }
}

fn mul_ternary_sparse<T: TensorData>(a: &Tensor<T>, b: &TernarySparseTensor) -> Tensor<T> {
    assert_eq!(a.shape.len(), 2);
    assert_eq!(b.shape.len(), 2);
    assert_eq!(a.shape[1], b.shape[0]);

    let mut data = vec![T::zero(); a.shape[0] * b.shape[1]];
    ark_std::cfg_iter!(b.pos).for_each(|idx| {
        let x = idx / b.shape[1];
        let y = idx % b.shape[1];
        ark_std::cfg_into_iter!(0..a.shape[0])
            .for_each(|k| data[k * b.shape[1] + y] += a.index(&[k, x]));
    });
    ark_std::cfg_iter!(b.neg).for_each(|idx| {
        let x = idx / b.shape[1];
        let y = idx % b.shape[1];
        ark_std::cfg_into_iter!(0..a.shape[0])
            .for_each(|k| data[k * b.shape[1] + y] -= a.index(&[k, x]));
    });

    Tensor {
        data,
        shape: vec![a.shape[0], b.shape[1]],
    }
}

impl<'a, T: Clone + Copy + Into<F>, F: JoltField> From<&'a Tensor<T>>
    for DenseMultilinearExtension<F>
{
    fn from(t: &'a Tensor<T>) -> Self {
        let vec = t.data.iter().map(|d| (*d).into()).collect_vec();
        DenseMultilinearExtension::from_evaluations_vec(t.num_vars(), vec)
    }
}

impl<'a, T: Clone + Copy + Into<F>, F: JoltField> From<&'a SparseTensor<T>>
    for DenseMultilinearExtension<F>
{
    fn from(t: &'a SparseTensor<T>) -> Self {
        let mut vec = vec![F::zero(); 1 << t.num_vars()];
        for (idx, value) in &t.data {
            vec[*idx] = (*value).into();
        }
        DenseMultilinearExtension::from_evaluations_vec(t.num_vars(), vec)
    }
}

impl<'a, T: Clone + Copy + Into<F>, F: JoltField> From<&'a SparseTensor<T>>
    for MySparseMultilinearExtension<F>
{
    fn from(t: &'a SparseTensor<T>) -> Self {
        let mut evaluations = Vec::with_capacity(t.data.len());
        for (idx, value) in &t.data {
            evaluations.push((*idx, (*value).into()));
        }
        MySparseMultilinearExtension::from_evaluations_vec(t.num_vars(), evaluations)
    }
}

impl<'a, F: JoltField> From<&'a TernarySparseTensor> for TernarySparseMultilinearExtension<F> {
    fn from(t: &'a TernarySparseTensor) -> Self {
        TernarySparseMultilinearExtension::from_evaluations(
            t.num_vars(),
            t.neg.clone(),
            t.pos.clone(),
        )
    }
}

impl<T> Tensors<T> {
    pub fn into_poly<F: JoltField>(self) -> DenseMultilinearExtension<F>
    where
        T: Clone + Copy + Into<F>,
    {
        let num_vars = self.num_vars();
        let data: Vec<Vec<F>> = self
            .0
            .into_iter()
            .map(|input| input.data.into_iter().map(|x| x.into()).collect_vec())
            .collect_vec();
        let data = transpose(data);
        let data = data.into_iter().flatten().collect_vec();
        assert_eq!(num_vars, data.len().log_2());
        DenseMultilinearExtension::from_evaluations_vec(num_vars, data)
    }
}

impl<'a, 'b, T: TensorData> Add<&'b Tensors<T>> for &'a Tensors<T> {
    type Output = Tensors<T>;

    fn add(self, other: &'b Tensors<T>) -> Tensors<T> {
        Tensors(
            izip!(&self.0, &other.0)
                .map(|(this, other)| this + other)
                .collect_vec(),
        )
    }
}

impl<T: TensorData> Add<Tensors<T>> for Tensors<T> {
    type Output = Tensors<T>;

    fn add(self, other: Tensors<T>) -> Tensors<T> {
        &self + &other
    }
}

impl<'a, T: TensorData> Add<Tensors<T>> for &'a Tensors<T> {
    type Output = Tensors<T>;

    fn add(self, other: Tensors<T>) -> Tensors<T> {
        self + &other
    }
}

impl<'b, T: TensorData> Add<&'b Tensors<T>> for Tensors<T> {
    type Output = Tensors<T>;

    fn add(self, other: &'b Tensors<T>) -> Tensors<T> {
        &self + other
    }
}

impl<'b, T: TensorData> Add<&'b Tensor<T>> for Tensors<T> {
    type Output = Tensors<T>;

    fn add(self, other: &'b Tensor<T>) -> Tensors<T> {
        Tensors(self.0.iter().map(|this| this + other).collect_vec())
    }
}

impl<'a, 'b, T: TensorData> Mul<&'b Tensor<T>> for &'a Tensors<T> {
    type Output = Tensors<T>;

    fn mul(self, other: &'b Tensor<T>) -> Tensors<T> {
        Tensors(self.0.iter().map(|this| this * other).collect_vec())
    }
}

impl<T: TensorData> Mul<Tensor<T>> for Tensors<T> {
    type Output = Tensors<T>;

    fn mul(self, other: Tensor<T>) -> Tensors<T> {
        &self * &other
    }
}

impl<'a, T: TensorData> Mul<Tensor<T>> for &'a Tensors<T> {
    type Output = Tensors<T>;

    fn mul(self, other: Tensor<T>) -> Tensors<T> {
        self * &other
    }
}

impl<'b, T: TensorData> Mul<&'b Tensor<T>> for Tensors<T> {
    type Output = Tensors<T>;

    fn mul(self, other: &'b Tensor<T>) -> Tensors<T> {
        &self * other
    }
}

impl<'a, 'b, T: TensorData> Mul<&'b SparseTensor<T>> for &'a Tensors<T> {
    type Output = Tensors<T>;

    fn mul(self, other: &'b SparseTensor<T>) -> Tensors<T> {
        Tensors(self.0.iter().map(|this| this * other).collect_vec())
    }
}

impl<'b, T: TensorData> Mul<&'b SparseTensor<T>> for Tensors<T> {
    type Output = Tensors<T>;

    fn mul(self, other: &'b SparseTensor<T>) -> Tensors<T> {
        &self * other
    }
}

impl<'a, 'b, T: TensorData> Mul<&'b TernarySparseTensor> for &'a Tensors<T> {
    type Output = Tensors<T>;

    fn mul(self, other: &'b TernarySparseTensor) -> Tensors<T> {
        Tensors(self.0.iter().map(|this| this * other).collect_vec())
    }
}

impl<'b, T: TensorData> Mul<&'b TernarySparseTensor> for Tensors<T> {
    type Output = Tensors<T>;

    fn mul(self, other: &'b TernarySparseTensor) -> Tensors<T> {
        &self * other
    }
}

impl TensorData for i64 {
    const BIT_WIDTH: usize = 64;
    const PRECISION: usize = 16;
    const RANGE_BIT_WIDTH: usize = 16;

    fn random<R: RngCore>(rng: &mut R) -> Self {
        rng.gen::<i16>() as i64
    }

    fn to_range_vector(&self) -> Vec<i64> {
        let tmp = self.abs();
        let m = (1 << Self::RANGE_BIT_WIDTH) - 1;
        vec![
            tmp & m,
            (tmp >> Self::RANGE_BIT_WIDTH) & m,
            (tmp >> (2 * Self::RANGE_BIT_WIDTH)) & m,
            (tmp >> (3 * Self::RANGE_BIT_WIDTH)) & m,
        ]
    }

    fn scaled_by(&self, n_bit: usize) -> Self {
        self << n_bit
    }

    fn divided_by(&self, n_bit: usize) -> Self {
        self >> n_bit
    }

    fn max(&self, other: &Self) -> Self {
        std::cmp::max(*self, *other)
    }
}

pub fn transpose<F: Clone + Copy>(matrix: Vec<Vec<F>>) -> Vec<Vec<F>> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut transpose = vec![vec![]; cols];

    ark_std::cfg_iter_mut!(transpose)
        .enumerate()
        .for_each(|(i, row)| {
            *row = Vec::with_capacity(rows);
            for matrix_j in matrix.iter() {
                row.push(matrix_j[i]);
            }
        });

    transpose
}
