use std::{fmt::Debug, rc::Rc};

use ark_std::{end_timer, start_timer, One, Zero};
use pcs::{
    commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme},
    poly::{dense::DenseMultilinearExtension, ternary_sparse::TernarySparseMultilinearExtension},
    utils::transcript::ProofTranscript,
};
use sumcheck::{protocol::ListOfProductsOfPolynomials, MLSumcheck};

use crate::{
    error::ZKMLError,
    lookup::LookupWitness,
    tensor::{Tensor, TensorData, Tensors, TernarySparseTensor},
};

use super::{
    linear::{LinearProof, LinearVerifier},
    LayerGenerator, LayerProver, LayerProverOutput, LayerVerifier, LayerWitness,
};

/// Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
#[derive(Clone, Debug)]
pub struct TernarySparseLinear<T, PCS> {
    pub transposed_weight: TernarySparseTensor,
    pub scaled_bias: Tensor<T>,
    pub has_bias: bool,
    _pcs: std::marker::PhantomData<PCS>,
}

impl<T: TensorData + Into<PCS::Field>, PCS: CommitmentScheme> TernarySparseLinear<T, PCS> {
    pub fn new(
        transposed_weight: TernarySparseTensor,
        scaled_bias: Tensor<T>,
        has_bias: bool,
    ) -> Self {
        TernarySparseLinear {
            transposed_weight,
            scaled_bias,
            has_bias,
            _pcs: std::marker::PhantomData,
        }
    }
}

impl<T: TensorData + Into<PCS::Field>, PCS: CommitmentScheme> LayerGenerator<T, PCS>
    for TernarySparseLinear<T, PCS>
{
    fn prover_and_verifier(&self) -> (Box<dyn LayerProver<PCS>>, Box<dyn LayerVerifier<PCS>>) {
        let TernarySparseLinear {
            transposed_weight,
            scaled_bias,
            has_bias,
            ..
        } = self;
        let transposed_weight = TernarySparseMultilinearExtension::from(transposed_weight);
        let scaled_bias = Rc::new(DenseMultilinearExtension::from(scaled_bias));
        let dense_transposed_weight = Rc::new(transposed_weight.to_dense_multilinear_extension());
        let srs = [
            PCS::setup(&[CommitShape::new(
                1 << dense_transposed_weight.num_vars(),
                BatchType::Small,
            )]),
            PCS::setup(&[CommitShape::new(
                1 << scaled_bias.num_vars(),
                BatchType::Small,
            )]),
        ];
        let model_comm = [
            PCS::commit(&dense_transposed_weight, &srs[0]),
            PCS::commit(&scaled_bias, &srs[1]),
        ];

        (
            Box::new(TernarySparseLinearProver {
                srs: srs.clone(),
                transposed_weight,
                dense_transposed_weight,
                scaled_bias,
                has_bias: *has_bias,
                _pcs: std::marker::PhantomData,
            }),
            Box::new(LinearVerifier {
                srs,
                model_comm,
                has_bias: self.has_bias,
            }),
        )
    }

    fn forward(
        &self,
        input: &Tensors<T>,
    ) -> (
        Tensors<T>,
        LayerWitness<PCS::Field>,
        LookupWitness<PCS::Field>,
    ) {
        (
            input * &self.transposed_weight + &self.scaled_bias,
            LayerWitness::default(),
            LookupWitness::default(),
        )
    }
}

pub struct TernarySparseLinearProver<PCS: CommitmentScheme> {
    pub srs: [PCS::Setup; 2],
    pub transposed_weight: TernarySparseMultilinearExtension<PCS::Field>,
    pub dense_transposed_weight: Rc<DenseMultilinearExtension<PCS::Field>>,
    pub scaled_bias: Rc<DenseMultilinearExtension<PCS::Field>>,
    pub has_bias: bool,
    pub _pcs: std::marker::PhantomData<PCS::Field>,
}

impl<PCS: CommitmentScheme> TernarySparseLinearProver<PCS> {
    pub fn setup_bias_value(&self, out_point: &[PCS::Field]) -> PCS::Field {
        if self.has_bias {
            self.scaled_bias.evaluate(out_point)
        } else {
            PCS::Field::zero()
        }
    }

    pub fn setup(
        &self,
        inputs: DenseMultilinearExtension<PCS::Field>,
        lo_point: &[PCS::Field],
        hi_point: &[PCS::Field],
    ) -> ListOfProductsOfPolynomials<PCS::Field> {
        let input_poly = Rc::new(inputs.fix_variables(lo_point));
        let weight_poly = Rc::new(
            self.transposed_weight
                .fix_variables(hi_point)
                .to_dense_multilinear_extension(),
        );
        let mut identity = ListOfProductsOfPolynomials::new(input_poly.num_vars);
        identity.add_product([input_poly, weight_poly], PCS::Field::one());
        identity
    }

    #[inline]
    pub fn input_idx() -> usize {
        0
    }

    #[inline]
    pub fn weight_idx() -> usize {
        1
    }
}

impl<PCS: CommitmentScheme> LayerProver<PCS> for TernarySparseLinearProver<PCS> {
    fn prove(
        &self,
        inputs: DenseMultilinearExtension<PCS::Field>,
        _witness: LayerWitness<PCS::Field>,
        lo_point: &[PCS::Field],
        hi_point: &[PCS::Field],
        _lookup_point: &[PCS::Field],
        transcript: &mut ProofTranscript,
    ) -> Result<LayerProverOutput<PCS>, ZKMLError> {
        let timer = start_timer!(|| "TernaryProver::prove");
        let identity = self.setup(inputs, lo_point, hi_point);

        let (sc_proof, state) = MLSumcheck::prove_as_subprotocol(transcript, &identity)?;
        let mat_point = [hi_point.to_vec(), state.randomness.clone()].concat();
        let openings = [
            PCS::prove(
                &self.srs[0],
                &self.dense_transposed_weight,
                &mat_point,
                transcript,
            ),
            PCS::prove(&self.srs[1], &self.scaled_bias, hi_point, transcript),
        ];

        let proof = LinearProof {
            sc_proof,
            input: state.final_evaluation(Self::input_idx()),
            weight: state.final_evaluation(Self::weight_idx()),
            bias: self.setup_bias_value(hi_point),
            openings,
        };
        transcript.append_scalar(&proof.input);
        end_timer!(timer);

        Ok(LayerProverOutput {
            lo_point: lo_point.to_vec(),
            hi_point: state.randomness,
            proof: proof.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Fr, G1Projective};
    use itertools::Itertools;
    use pcs::{
        commitment::{commitment_scheme::CommitmentScheme, hyrax::HyraxScheme},
        field::JoltField,
        poly::{
            dense::DenseMultilinearExtension, ternary_sparse::TernarySparseMultilinearExtension,
        },
        utils::transcript::ProofTranscript,
    };
    use rand::{prelude::SliceRandom, Rng, RngCore};

    use crate::{
        layers::{
            ternary_sparse_linear::TernarySparseLinear, LayerGenerator, LayerProverOutput,
            LayerWitness,
        },
        tensor::{Tensor, TensorClaim, TensorData, Tensors, TernarySparseTensor},
    };

    type Hyrax = HyraxScheme<G1Projective>;

    struct TernarySparseLinearInstance<T, PCS: CommitmentScheme> {
        ternary_sparse_linear: TernarySparseLinear<T, PCS>,
        inputs: DenseMultilinearExtension<PCS::Field>,
        outputs: DenseMultilinearExtension<PCS::Field>,
        instance_num_vars: usize,
    }

    fn test_helper<T: TensorData + Into<PCS::Field>, PCS: CommitmentScheme, R: RngCore>(
        mut random: impl FnMut(&mut R) -> T,
        rng: &mut R,
    ) -> TernarySparseLinearInstance<T, PCS> {
        let mut idx = (0..8usize).collect_vec();
        idx.shuffle(rng);
        let transposed_weight = {
            let rand_poly =
                TernarySparseMultilinearExtension::<PCS::Field>::rand_with_config(3, 4, false, rng);
            TernarySparseTensor {
                neg: rand_poly.evaluations_neg,
                pos: rand_poly.evaluations_pos,
                shape: vec![4, 2],
            }
        };
        let bias = Tensor {
            data: (0..2).map(|_| random(rng)).collect(),
            shape: vec![1, 2],
        };
        let inputs = Tensors(
            (0..4)
                .map(|_| Tensor {
                    data: (0..4).map(|_| random(rng)).collect(),
                    shape: vec![1, 4],
                })
                .collect_vec(),
        );

        let dense_tensor = {
            let mut evaluations = vec![T::zero(); 8];
            for &i in transposed_weight.neg.iter() {
                evaluations[i] = -T::one();
            }
            for &i in transposed_weight.pos.iter() {
                evaluations[i] = T::one();
            }
            evaluations
        };
        let result = Tensors(
            inputs
                .0
                .iter()
                .map(|input| {
                    Tensor::new(
                        vec![
                            input.data[0] * dense_tensor[0]
                                + input.data[1] * dense_tensor[2]
                                + input.data[2] * dense_tensor[4]
                                + input.data[3] * dense_tensor[6]
                                + bias.data[0],
                            input.data[0] * dense_tensor[1]
                                + input.data[1] * dense_tensor[3]
                                + input.data[2] * dense_tensor[5]
                                + input.data[3] * dense_tensor[7]
                                + bias.data[1],
                        ],
                        vec![1, 2],
                    )
                })
                .collect_vec(),
        );

        let ternary_sparse_linear =
            TernarySparseLinear::<T, PCS>::new(transposed_weight, bias, true);
        let (outputs, _, _) = ternary_sparse_linear.forward(&inputs);

        assert_eq!(outputs, result);
        let inputs = inputs.into_poly();
        let outputs = outputs.into_poly();
        TernarySparseLinearInstance {
            ternary_sparse_linear,
            inputs,
            outputs,
            instance_num_vars: 2,
        }
    }

    #[test]
    fn test_forward() {
        let mut rng = rand::thread_rng();

        // can't be overflow
        let _ = test_helper::<i64, Hyrax, _>(|rng| rng.gen::<i32>() as i64, &mut rng);
    }
    #[test]
    fn test_prove_and_verify() {
        let mut rng = rand::thread_rng();

        let TernarySparseLinearInstance {
            ternary_sparse_linear,
            inputs,
            outputs,
            instance_num_vars,
        } = test_helper::<i64, Hyrax, _>(|rng| rng.gen::<i32>() as i64, &mut rng);

        let out_point = (0..3).map(|_| Fr::random(&mut rng)).collect_vec();
        let (lo_point, hi_point) = out_point.split_at(instance_num_vars);
        let out_evaluation = outputs.evaluate(&out_point);
        let (prover, verifier) = ternary_sparse_linear.prover_and_verifier();
        let proof = {
            let mut transcript = ProofTranscript::new(b"test");
            let LayerProverOutput { proof, .. } = prover
                .prove(
                    inputs,
                    LayerWitness::default(),
                    lo_point,
                    hi_point,
                    &[],
                    &mut transcript,
                )
                .expect("Prover failed");
            proof
        };

        let _ = {
            let mut transcript = ProofTranscript::new(b"test");
            let output_claim = TensorClaim {
                lo_point: lo_point.to_vec(),
                hi_point: hi_point.to_vec(),
                value: out_evaluation,
            };
            verifier
                .verify(&output_claim, &None, proof, &mut transcript)
                .expect("Verifier failed")
        };
    }
}
