use std::{fmt::Debug, rc::Rc};

use ark_std::{end_timer, start_timer, One, Zero};
use itertools::{izip, Itertools};
use pcs::{
    commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme},
    field::JoltField,
    poly::dense::DenseMultilinearExtension,
    utils::transcript::ProofTranscript,
};
use sumcheck::{
    protocol::{ListOfProductsOfPolynomials, PolynomialInfo},
    MLSumcheck,
};

use crate::{
    error::ZKMLError,
    lookup::{LookupClaim, LookupWitness},
    tensor::{Tensor, TensorClaim, TensorData, Tensors},
};

use super::{
    LayerClaim, LayerGenerator, LayerProof, LayerProver, LayerProverOutput, LayerVerifier,
    LayerWitness,
};

type SumcheckProof<F> = sumcheck::Proof<F>;

#[derive(Clone, Debug)]
pub struct Linear<T, PCS> {
    pub transposed_weight: Tensor<T>,
    pub scaled_bias: Tensor<T>,
    pub has_bias: bool,
    pub _pcs: std::marker::PhantomData<PCS>,
}

impl<T: TensorData, PCS: CommitmentScheme> Linear<T, PCS> {
    pub fn new(transposed_weight: Tensor<T>, bias: Tensor<T>, has_bias: bool) -> Self {
        Linear {
            transposed_weight,
            scaled_bias: bias,
            has_bias,
            _pcs: std::marker::PhantomData,
        }
    }
}

impl<T: TensorData + Into<PCS::Field>, PCS: CommitmentScheme> LayerGenerator<T, PCS>
    for Linear<T, PCS>
{
    fn forward(
        &self,
        inputs: &Tensors<T>,
    ) -> (
        Tensors<T>,
        LayerWitness<PCS::Field>,
        LookupWitness<PCS::Field>,
    ) {
        (
            Tensors(
                inputs
                    .0
                    .iter()
                    .map(|input| input * &self.transposed_weight + &self.scaled_bias)
                    .collect_vec(),
            ),
            LayerWitness::default(),
            LookupWitness::default(),
        )
    }

    fn prover_and_verifier(&self) -> (Box<dyn LayerProver<PCS>>, Box<dyn LayerVerifier<PCS>>) {
        let Linear {
            transposed_weight,
            scaled_bias,
            has_bias,
            ..
        } = self;
        let transposed_weight = Rc::new(DenseMultilinearExtension::from(transposed_weight));
        let scaled_bias = Rc::new(DenseMultilinearExtension::from(scaled_bias));
        let srs = [
            PCS::setup(&[CommitShape::new(
                1 << transposed_weight.num_vars(),
                BatchType::Small,
            )]),
            PCS::setup(&[CommitShape::new(
                1 << scaled_bias.num_vars(),
                BatchType::Small,
            )]),
        ];
        let model_comm = [
            PCS::commit(&transposed_weight, &srs[0]),
            PCS::commit(&scaled_bias, &srs[1]),
        ];

        (
            Box::new(LinearProver {
                srs: srs.clone(),
                transposed_weight,
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
}

/// Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
#[derive(Clone, Debug)]
pub struct LinearProver<PCS: CommitmentScheme> {
    pub srs: [PCS::Setup; 2],
    pub transposed_weight: Rc<DenseMultilinearExtension<PCS::Field>>,
    pub scaled_bias: Rc<DenseMultilinearExtension<PCS::Field>>,
    pub has_bias: bool,
    pub _pcs: std::marker::PhantomData<PCS>,
}

#[derive(Clone, Debug, Default)]
pub struct LinearWitness;

#[derive(Clone, Debug)]
pub struct LinearProof<PCS: CommitmentScheme> {
    pub sc_proof: SumcheckProof<PCS::Field>,
    pub weight: PCS::Field,
    pub bias: PCS::Field,
    pub input: PCS::Field,
    pub openings: [PCS::Proof; 2],
}

#[derive(Clone, Debug, Default)]
pub struct LinearClaim<F: JoltField> {
    _field: std::marker::PhantomData<F>,
}

impl<PCS: CommitmentScheme> LinearProver<PCS> {
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
        let weight_poly = Rc::new(self.transposed_weight.fix_variables(hi_point));
        let num_vars = input_poly.num_vars();
        let mut identity = ListOfProductsOfPolynomials::new(num_vars);
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

impl<PCS: CommitmentScheme> LayerProver<PCS> for LinearProver<PCS> {
    fn prove(
        &self,
        inputs: DenseMultilinearExtension<PCS::Field>,
        _witness: LayerWitness<PCS::Field>,
        lo_point: &[PCS::Field],
        hi_point: &[PCS::Field],
        _lookup_point: &[PCS::Field],
        transcript: &mut ProofTranscript,
    ) -> Result<LayerProverOutput<PCS>, ZKMLError> {
        let timer = start_timer!(|| "LinearProver::prove");
        let identity = self.setup(inputs, lo_point, hi_point);

        let (sc_proof, state) = MLSumcheck::prove_as_subprotocol(transcript, &identity)?;

        let mat_point = [hi_point.to_vec(), state.randomness.clone()].concat();
        let openings = [
            PCS::prove(
                &self.srs[0],
                &self.transposed_weight,
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
            proof: proof.into(),
            lo_point: lo_point.to_vec(),
            hi_point: state.randomness,
        })
    }
}

#[derive(Clone, Debug)]
pub struct LinearVerifier<PCS: CommitmentScheme> {
    pub srs: [PCS::Setup; 2],
    pub model_comm: [PCS::Commitment; 2],
    pub has_bias: bool,
}

impl<PCS: CommitmentScheme> LinearVerifier<PCS> {
    pub fn sum(out_eval: PCS::Field, proof: &LinearProof<PCS>) -> PCS::Field {
        out_eval - proof.bias
    }

    pub fn final_evaluation(proof: &LinearProof<PCS>) -> PCS::Field {
        proof.weight * proof.input
    }
}

impl<PCS: CommitmentScheme> LayerVerifier<PCS> for LinearVerifier<PCS> {
    fn verify(
        &self,
        output_claim: &TensorClaim<PCS::Field>,
        _lookup_claim: &Option<LookupClaim<PCS::Field>>,
        proof: LayerProof<PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(TensorClaim<PCS::Field>, Vec<LayerClaim<PCS::Field>>), ZKMLError> {
        let TensorClaim {
            lo_point,
            hi_point,
            value,
        } = output_claim;

        let proof: LinearProof<PCS> = proof.try_into()?;
        let identity_info = PolynomialInfo {
            max_multiplicands: 2,
            num_variables: proof.sc_proof.len(),
        };

        let claim = MLSumcheck::verify_as_subprotocol(
            transcript,
            &identity_info,
            Self::sum(*value, &proof),
            &proof.sc_proof,
        )?;

        if claim.expected_evaluation != Self::final_evaluation(&proof) {
            return Err(ZKMLError::SumcheckClaim(stringify!(Linear)));
        }

        let model_evals = [&proof.weight, &proof.bias];
        let mat_point = [hi_point.to_vec(), claim.point.clone()].concat();
        let points = [&mat_point, hi_point];
        for (opening, srs, eval, comm, point) in izip!(
            &proof.openings,
            &self.srs,
            model_evals,
            &self.model_comm,
            &points
        ) {
            PCS::verify(opening, srs, transcript, point, eval, comm)?;
        }

        transcript.append_scalar(&proof.input);

        Ok((
            TensorClaim {
                lo_point: lo_point.clone(),
                hi_point: claim.point,
                value: proof.input,
            },
            vec![],
        ))
    }
}

impl<F: JoltField> TryFrom<LayerWitness<F>> for LinearWitness {
    type Error = ZKMLError;
    fn try_from(wit: LayerWitness<F>) -> Result<Self, Self::Error> {
        if !wit.0.is_empty() {
            return Err(ZKMLError::InvalidNodeWitness(stringify!(Linear)));
        }
        Ok(LinearWitness {})
    }
}

impl<F: JoltField> From<LinearWitness> for LayerWitness<F> {
    fn from(_: LinearWitness) -> Self {
        LayerWitness::default()
    }
}

impl<PCS: CommitmentScheme> TryFrom<LayerProof<PCS>> for LinearProof<PCS> {
    type Error = ZKMLError;
    fn try_from(proof: LayerProof<PCS>) -> Result<Self, Self::Error> {
        if proof.claims.len() != 3 {
            return Err(ZKMLError::InvalidLayerClaims(
                stringify!(Linear),
                3,
                proof.claims.len(),
            ));
        }
        Ok(LinearProof {
            sc_proof: proof.sc_proof,
            weight: proof.claims[0],
            bias: proof.claims[1],
            input: proof.claims[2],
            openings: proof
                .model_openings
                .try_into()
                .or(Err(ZKMLError::InvalidNodeProof(
                    "wrong linear layer openings",
                )))?,
        })
    }
}

impl<PCS: CommitmentScheme> From<LinearProof<PCS>> for LayerProof<PCS> {
    fn from(val: LinearProof<PCS>) -> Self {
        LayerProof {
            sc_proof: val.sc_proof,
            claims: vec![val.weight, val.bias, val.input],
            model_openings: val.openings.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Fr, G1Projective};
    use itertools::Itertools;
    use pcs::{
        commitment::{commitment_scheme::CommitmentScheme, hyrax::HyraxScheme},
        field::JoltField,
        poly::dense::DenseMultilinearExtension,
        utils::transcript::ProofTranscript,
    };
    use rand::Rng;

    use crate::{
        layers::{LayerGenerator, LayerProverOutput},
        tensor::{Tensor, TensorClaim, TensorData, Tensors},
    };

    use super::{Linear, LinearWitness};

    type Hyrax = HyraxScheme<G1Projective>;

    struct LinearInstance<T: TensorData, PCS: CommitmentScheme> {
        linear: Linear<T, PCS>,
        inputs: DenseMultilinearExtension<PCS::Field>,
        outputs: DenseMultilinearExtension<PCS::Field>,
        instance_num_vars: usize,
    }

    fn test_helper<T: TensorData + Into<PCS::Field>, PCS: CommitmentScheme>(
        mut random: impl FnMut() -> T,
        instance_num_vars: usize,
    ) -> LinearInstance<T, PCS> {
        let transposed_weight = Tensor {
            data: (0..8).map(|_| random()).collect(),
            shape: vec![4, 2],
        };
        let bias = Tensor {
            data: (0..2).map(|_| random()).collect(),
            shape: vec![1, 2],
        };
        let inputs = Tensors(
            (0..1 << instance_num_vars)
                .map(|_| Tensor {
                    data: (0..4).map(|_| random()).collect(),
                    shape: vec![1, 4],
                })
                .collect_vec(),
        );
        let result = Tensors(
            inputs
                .0
                .iter()
                .map(|input| {
                    Tensor::new(
                        vec![
                            input.data[0] * transposed_weight.data[0]
                                + input.data[1] * transposed_weight.data[2]
                                + input.data[2] * transposed_weight.data[4]
                                + input.data[3] * transposed_weight.data[6]
                                + bias.data[0],
                            input.data[0] * transposed_weight.data[1]
                                + input.data[1] * transposed_weight.data[3]
                                + input.data[2] * transposed_weight.data[5]
                                + input.data[3] * transposed_weight.data[7]
                                + bias.data[1],
                        ],
                        vec![1, 2],
                    )
                })
                .collect_vec(),
        );

        let linear = Linear::<T, PCS>::new(transposed_weight, bias, true);
        let (outputs, _, _) = linear.forward(&inputs);

        assert_eq!(outputs, result);
        let inputs = inputs.into_poly();
        let outputs = outputs.into_poly();
        LinearInstance {
            linear,
            inputs,
            outputs,
            instance_num_vars,
        }
    }

    #[test]
    fn test_forward() {
        let mut rng = rand::thread_rng();

        // can't be overflow
        let _ = test_helper::<i64, Hyrax>(|| rng.gen::<i32>() as i64, 2);
    }

    #[test]
    fn test_prove_and_verify() {
        let mut rng = rand::thread_rng();

        let LinearInstance {
            linear,
            inputs,
            outputs,
            instance_num_vars,
        } = test_helper::<i64, Hyrax>(|| rng.gen::<i32>() as i64, 2);

        let out_point = (0..outputs.num_vars())
            .map(|_| Fr::random(&mut rng))
            .collect_vec();
        let out_evaluation = outputs.evaluate(&out_point);
        let (lo_point, hi_point) = out_point.split_at(instance_num_vars);
        let (prover, verifier) = linear.prover_and_verifier();
        let proof = {
            let mut transcript = ProofTranscript::new(b"test");
            let LayerProverOutput { proof, .. } = prover
                .prove(
                    inputs,
                    LinearWitness.into(),
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
