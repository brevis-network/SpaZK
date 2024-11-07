use std::{fmt::Debug, rc::Rc};

use ark_std::{end_timer, start_timer, One, Zero};
use itertools::{izip, Itertools};
use pcs::{
    commitment::commitment_scheme::CommitmentScheme,
    field::JoltField,
    poly::{dense::DenseMultilinearExtension, eq_poly::EqPolynomial},
    utils::transcript::ProofTranscript,
};
use sumcheck::{
    protocol::{ListOfProductsOfPolynomials, PolynomialInfo},
    MLSumcheck,
};

use crate::{
    error::ZKMLError,
    lookup::{LookupClaim, LookupWitness},
    tensor::{TensorClaim, TensorData, Tensors},
};

use super::{
    LayerClaim, LayerGenerator, LayerProof, LayerProver, LayerProverOutput, LayerVerifier,
    LayerWitness,
};

type SumcheckProof<F> = sumcheck::Proof<F>;

/// Applies a relu transformation to the incoming data: :math:`x = max(x, 0) >> PRECISION`.
#[derive(Clone, Debug, Default)]
pub struct Relu<T, PCS> {
    should_rescale: bool,
    _tensor: std::marker::PhantomData<T>,
    _pcs: std::marker::PhantomData<PCS>,
}

#[derive(Clone, Debug)]
pub struct ReluWitness<F: JoltField> {
    is_positive: DenseMultilinearExtension<F>,
    in_decomp: Vec<DenseMultilinearExtension<F>>,
}

#[derive(Clone, Debug)]
pub struct ReluProof<PCS: CommitmentScheme> {
    pub sc_proof: SumcheckProof<PCS::Field>,
    pub in_decomp: Vec<PCS::Field>,
    pub is_positive: PCS::Field,
    pub input: PCS::Field,
}

#[derive(Clone, Debug)]
pub struct ReluClaim<F: JoltField> {
    lo_point: Vec<F>,
    hi_point: Vec<F>,
    in_decomp: Vec<F>,
    is_positive: F,
}

impl<T, PCS> Relu<T, PCS> {
    pub fn new(should_rescale: bool) -> Self {
        Relu {
            should_rescale,
            _tensor: std::marker::PhantomData,
            _pcs: std::marker::PhantomData,
        }
    }
}

impl<T: TensorData + Into<PCS::Field>, PCS: CommitmentScheme> LayerGenerator<T, PCS>
    for Relu<T, PCS>
{
    fn prover_and_verifier(&self) -> (Box<dyn LayerProver<PCS>>, Box<dyn LayerVerifier<PCS>>) {
        (
            Box::new(ReluProver {
                should_rescale: self.should_rescale,
                _pcs: std::marker::PhantomData,
                precision: T::PRECISION,
                range_bit_width: T::RANGE_BIT_WIDTH,
            }),
            Box::new(ReluVerifier {
                should_rescale: self.should_rescale,
                _pcs: std::marker::PhantomData,
                range_bit_width: T::RANGE_BIT_WIDTH,
                precision: T::PRECISION,
            }),
        )
    }

    fn forward(
        &self,
        inputs: &Tensors<T>,
    ) -> (
        Tensors<T>,
        LayerWitness<PCS::Field>,
        LookupWitness<PCS::Field>,
    ) {
        let is_positive = inputs.is_positive().into_poly();
        let outputs = if self.should_rescale {
            inputs.relu_and_rescale(T::PRECISION)
        } else {
            inputs.relu()
        };

        let in_decomp = inputs
            .to_range_vectors()
            .into_iter()
            .map(|vec| vec.into_poly())
            .collect_vec();

        let range = {
            let mut range = in_decomp
                .iter()
                .cloned()
                .flat_map(|vec| vec.evaluations)
                .collect_vec();
            range.resize(range.len().next_power_of_two(), PCS::Field::zero());
            range
        };

        (
            outputs,
            ReluWitness {
                is_positive,
                in_decomp,
            }
            .into(),
            LookupWitness { range },
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct ReluProver<PCS: CommitmentScheme> {
    _pcs: std::marker::PhantomData<PCS>,
    should_rescale: bool,
    precision: usize,
    range_bit_width: usize,
}

impl<PCS: CommitmentScheme> ReluProver<PCS> {
    pub fn setup(
        &self,
        inputs: DenseMultilinearExtension<PCS::Field>,
        witness: ReluWitness<PCS::Field>,
        lo_point: &[PCS::Field],
        hi_point: &[PCS::Field],
        lookup_point: &[PCS::Field],
        coeffs: &[PCS::Field],
    ) -> (
        ListOfProductsOfPolynomials<PCS::Field>,
        Vec<DenseMultilinearExtension<PCS::Field>>,
    ) {
        let num_vars = lo_point.len() + hi_point.len();
        let lo_eq = DenseMultilinearExtension::from_evaluations_vec(
            lo_point.len(),
            EqPolynomial::evals(lo_point),
        );
        let hi_eq = DenseMultilinearExtension::from_evaluations_vec(
            hi_point.len(),
            EqPolynomial::evals(hi_point),
        );
        let out_eq = hi_eq
            .iter()
            .flat_map(|hi| lo_eq.iter().map(|lo| *lo * hi).collect_vec())
            .collect_vec();
        let out_eq = DenseMultilinearExtension::from_evaluations_vec(num_vars, out_eq);
        let lookup_eq = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            EqPolynomial::evals(&lookup_point[..num_vars]),
        );

        let ReluWitness {
            is_positive,
            in_decomp,
        } = witness;

        let out_eq = Rc::new(out_eq);
        let lookup_eq = Rc::new(lookup_eq);
        let input = Rc::new(inputs);
        let is_positive = Rc::new(is_positive);

        let mut identity = ListOfProductsOfPolynomials::new(num_vars);
        let recovered_input = Rc::new(
            in_decomp
                .iter()
                .enumerate()
                .map(|(i, poly)| {
                    poly * &PCS::Field::from_u64(1u64 << (self.range_bit_width * i)).unwrap()
                })
                .reduce(|a, b| a + b)
                .unwrap_or_default(),
        );
        // 1. Compute the output: output = is_positive * (in_decomp[PRECISION / RANGE_BIT_WIDTH] * 2^{RANGE_BIT_WIDTH * 0} + ...)
        let rescaled_input = if self.should_rescale {
            Rc::new(
                in_decomp
                    .iter()
                    .skip(self.precision / self.range_bit_width)
                    .enumerate()
                    .map(|(i, poly)| {
                        poly * &PCS::Field::from_u64(1u64 << (self.range_bit_width * i)).unwrap()
                    })
                    .reduce(|a, b| a + b)
                    .unwrap_or_default(),
            )
        } else {
            recovered_input.clone()
        };

        identity.add_product(
            [out_eq.clone(), is_positive.clone(), rescaled_input],
            coeffs[0],
        );

        // 2. Check in_decomp can recover input
        //      0 = 2 * is_positive * input - input - (in_decomp[0] * 2^{RANGE_BIT_WIDTH * 0} + ...)

        identity.add_product(
            [out_eq.clone(), is_positive.clone(), input.clone()],
            coeffs[1] + coeffs[1],
        );
        identity.add_product([out_eq.clone(), input], -coeffs[1]);
        identity.add_product([out_eq.clone(), recovered_input], -coeffs[1]);

        // 3. Check is_positive are all bits.
        //      0 = is_positive * is_positive - is_positive
        identity.add_product(
            [out_eq.clone(), is_positive.clone(), is_positive.clone()],
            coeffs[2],
        );
        identity.add_product([out_eq, is_positive], -coeffs[2]);
        // 4. Rerandomize the lookup witness: range = lookup_eq * (lookup_coeff[0] * in_decomp[0] + ...)
        let lookup_coeffs = EqPolynomial::evals(&lookup_point[num_vars..]);
        let randomed_lookup_items = Rc::new(
            izip!(&in_decomp, &lookup_coeffs)
                .map(|(poly, lookup_coeff)| poly * lookup_coeff)
                .reduce(|a, b| a + b)
                .unwrap_or_default(),
        );
        identity.add_product([lookup_eq, randomed_lookup_items], coeffs[3]);

        (identity, in_decomp)
    }

    #[inline]
    pub fn input_idx(&self) -> usize {
        3
    }

    #[inline]
    pub fn is_positive_idx() -> usize {
        1
    }
}

impl<PCS: CommitmentScheme> LayerProver<PCS> for ReluProver<PCS> {
    fn prove(
        &self,
        inputs: DenseMultilinearExtension<PCS::Field>,
        witness: LayerWitness<PCS::Field>,
        lo_point: &[PCS::Field],
        hi_point: &[PCS::Field],
        lookup_point: &[PCS::Field],
        transcript: &mut ProofTranscript,
    ) -> Result<LayerProverOutput<PCS>, ZKMLError> {
        let timer = start_timer!(|| "ReluProver::prove");
        let coeffs = transcript.challenge_scalar_powers(4);

        let witness = witness.try_into()?;
        let (identity, in_decomp) =
            self.setup(inputs, witness, lo_point, hi_point, lookup_point, &coeffs);

        let (sc_proof, state) = MLSumcheck::prove_as_subprotocol(transcript, &identity)?;
        let proof = ReluProof {
            sc_proof,
            input: state.final_evaluation(self.input_idx()),
            is_positive: state.final_evaluation(Self::is_positive_idx()),
            in_decomp: in_decomp
                .into_iter()
                .map(|poly| poly.evaluate(&state.randomness))
                .collect_vec(),
        };
        transcript.append_scalar(&proof.input);

        let lo_num_vars = lo_point.len();
        end_timer!(timer);
        Ok(LayerProverOutput {
            proof: proof.into(),
            lo_point: state.randomness[..lo_num_vars].to_vec(),
            hi_point: state.randomness[lo_num_vars..].to_vec(),
        })
    }
}

pub struct ReluVerifier<PCS> {
    should_rescale: bool,
    range_bit_width: usize,
    precision: usize,
    _pcs: std::marker::PhantomData<PCS>,
}

impl<PCS: CommitmentScheme> ReluVerifier<PCS> {
    pub fn sum(
        &self,
        out_eval: PCS::Field,
        lookup_eval: PCS::Field,
        coeffs: &[PCS::Field],
    ) -> Result<PCS::Field, ZKMLError> {
        Ok(out_eval + coeffs[3] * lookup_eval)
    }

    pub fn final_evaluation(
        &self,
        proof: &ReluProof<PCS>,
        lo_point: &[PCS::Field],
        hi_point: &[PCS::Field],
        lookup_point: &[PCS::Field],
        claim_point: &[PCS::Field],
        coeffs: &[PCS::Field],
    ) -> PCS::Field {
        let one = PCS::Field::one();
        let num_vars = lo_point.len() + hi_point.len();
        let out_eq_value = EqPolynomial::new(lo_point.to_vec())
            .evaluate(&claim_point[..lo_point.len()])
            * EqPolynomial::new(hi_point.to_vec()).evaluate(&claim_point[lo_point.len()..]);
        let lookup_eq_value =
            EqPolynomial::new(lookup_point[..num_vars].to_vec()).evaluate(claim_point);

        let mut value = PCS::Field::zero();
        // 1. Compute the output: output = is_positive * * (in_decomp[PRECISION / RANGE_BIT_WIDTH] * 2^{RANGE_BIT_WIDTH * 0} + ...)
        let recovered_input = proof
            .in_decomp
            .iter()
            .enumerate()
            .map(|(i, eval)| {
                *eval * PCS::Field::from_u64(1u64 << (self.range_bit_width * i)).unwrap()
            })
            .reduce(|a, b| a + b)
            .unwrap_or_default();

        let rescaled_input = if self.should_rescale {
            proof
                .in_decomp
                .iter()
                .skip(self.precision / self.range_bit_width)
                .enumerate()
                .map(|(i, eval)| {
                    *eval * PCS::Field::from_u64(1u64 << (self.range_bit_width * i)).unwrap()
                })
                .reduce(|a, b| a + b)
                .unwrap_or_default()
        } else {
            recovered_input
        };
        value += coeffs[0] * out_eq_value * proof.is_positive * rescaled_input;

        // 2. Check in_decomp can recover input
        //      0 = 2 * is_positive * input - input - (in_decomp[0] * 2^{RANGE_BIT_WIDTH * 0} + ...)
        value += coeffs[1]
            * out_eq_value
            * ((proof.is_positive + proof.is_positive - one) * proof.input - recovered_input);

        // 3. Check is_positive are all bits.
        //      0 = (is_positive - 1) * is_positive
        value += coeffs[2] * out_eq_value * (proof.is_positive - one) * proof.is_positive;

        // 4. Rerandomize the lookup witness: range = lookup_eq * (lookup_coeff[0] * in_decomp[0] + ...)
        let lookup_coeffs = EqPolynomial::evals(&lookup_point[num_vars..]);
        let randomed_lookup_items: PCS::Field = izip!(&proof.in_decomp, &lookup_coeffs)
            .map(|(in_decomp_eval, lookup_coeff_eval)| *in_decomp_eval * lookup_coeff_eval)
            .sum();
        value += coeffs[3] * lookup_eq_value * randomed_lookup_items;
        value
    }
}

impl<PCS: CommitmentScheme> LayerVerifier<PCS> for ReluVerifier<PCS> {
    fn verify(
        &self,
        output_claim: &TensorClaim<PCS::Field>,
        lookup_claim: &Option<LookupClaim<PCS::Field>>,
        proof: LayerProof<PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(TensorClaim<PCS::Field>, Vec<LayerClaim<PCS::Field>>), ZKMLError> {
        let coeffs = transcript.challenge_scalar_powers(4);

        let identity_info = PolynomialInfo {
            max_multiplicands: 3,
            num_variables: proof.sc_proof.len(),
        };

        let TensorClaim {
            lo_point,
            hi_point,
            value: output_value,
        } = output_claim;
        let LookupClaim {
            point: lookup_point,
            range,
        } = lookup_claim.as_ref().unwrap();
        let proof: ReluProof<PCS> = proof.try_into()?;

        let sum = self.sum(*output_value, *range, &coeffs)?;
        let claim =
            MLSumcheck::verify_as_subprotocol(transcript, &identity_info, sum, &proof.sc_proof)?;

        if claim.expected_evaluation
            != self.final_evaluation(
                &proof,
                lo_point,
                hi_point,
                lookup_point,
                &claim.point,
                &coeffs,
            )
        {
            return Err(ZKMLError::SumcheckClaim(concat!(
                stringify!(Relu),
                "/final-eval"
            )));
        }

        transcript.append_scalar(&proof.input);

        Ok((
            TensorClaim {
                lo_point: claim.point[..lo_point.len()].to_vec(),
                hi_point: claim.point[lo_point.len()..].to_vec(),
                value: proof.input,
            },
            ReluClaim {
                lo_point: claim.point[..lo_point.len()].to_vec(),
                hi_point: claim.point[lo_point.len()..].to_vec(),
                is_positive: proof.is_positive,
                in_decomp: proof.in_decomp,
            }
            .into(),
        ))
    }
}

impl<F: JoltField> TryFrom<LayerWitness<F>> for ReluWitness<F> {
    type Error = ZKMLError;

    fn try_from(mut witness: LayerWitness<F>) -> Result<Self, Self::Error> {
        if witness.0.is_empty() {
            return Err(ZKMLError::InvalidNodeWitness(stringify!(Relu)));
        }

        let is_positive = witness.0.pop().unwrap();
        Ok(ReluWitness {
            in_decomp: witness.0,
            is_positive,
        })
    }
}

impl<F: JoltField> From<ReluWitness<F>> for LayerWitness<F> {
    fn from(val: ReluWitness<F>) -> Self {
        let ReluWitness {
            in_decomp,
            is_positive,
        } = val;
        let mut wits = in_decomp;
        wits.push(is_positive);
        LayerWitness(wits)
    }
}

impl<PCS: CommitmentScheme> TryFrom<LayerProof<PCS>> for ReluProof<PCS> {
    type Error = ZKMLError;

    fn try_from(proof: LayerProof<PCS>) -> Result<Self, Self::Error> {
        if proof.claims.len() < 2 {
            return Err(ZKMLError::InvalidLayerClaims(
                stringify!(Relu),
                2,
                proof.claims.len(),
            ));
        }

        let LayerProof {
            sc_proof,
            mut claims,
            ..
        } = proof;
        let input = claims.pop().unwrap();
        let is_positive = claims.pop().unwrap();
        Ok(ReluProof {
            sc_proof,
            in_decomp: claims,
            is_positive,
            input,
        })
    }
}

impl<PCS: CommitmentScheme> From<ReluProof<PCS>> for LayerProof<PCS> {
    fn from(val: ReluProof<PCS>) -> Self {
        let ReluProof {
            sc_proof,
            in_decomp,
            is_positive,
            input,
        } = val;
        let mut claims = in_decomp;
        claims.extend([is_positive, input]);
        LayerProof {
            sc_proof,
            claims,
            model_openings: vec![],
        }
    }
}

impl<F: JoltField> From<ReluClaim<F>> for Vec<LayerClaim<F>> {
    fn from(val: ReluClaim<F>) -> Self {
        let ReluClaim {
            lo_point,
            hi_point,
            in_decomp,
            is_positive,
        } = val;
        let mut claims = in_decomp
            .into_iter()
            .map(|d| LayerClaim {
                lo_point: lo_point.clone(),
                hi_point: hi_point.clone(),
                value: d,
            })
            .collect_vec();
        claims.push(LayerClaim {
            lo_point,
            hi_point,
            value: is_positive,
        });
        claims
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Fr, G1Projective};
    use itertools::Itertools;
    use pcs::{
        commitment::{commitment_scheme::CommitmentScheme, hyrax::HyraxScheme},
        field::JoltField,
        field_vec,
        poly::dense::DenseMultilinearExtension,
        utils::{math::Math, transcript::ProofTranscript},
    };

    use crate::{
        layers::{LayerGenerator, LayerProverOutput},
        lookup::{LookupClaim, LookupWitness},
        tensor::{Tensor, TensorClaim, TensorData, Tensors},
    };

    use super::{Relu, ReluWitness};

    type Hyrax = HyraxScheme<G1Projective>;

    struct ReluInstance<T: TensorData, PCS: CommitmentScheme> {
        relu: Relu<T, PCS>,
        inputs: DenseMultilinearExtension<PCS::Field>,
        outputs: DenseMultilinearExtension<PCS::Field>,
        instance_num_vars: usize,

        relu_witness: ReluWitness<PCS::Field>,
        lookup_witness: LookupWitness<PCS::Field>,
    }

    fn test_helper(should_rescale: bool) -> ReluInstance<i64, Hyrax> {
        let inputs = Tensors(vec![
            Tensor {
                data: vec![
                    1,
                    -((3 << 16) | 5),
                    (7 << 32) | (11 << 16) | 13,
                    -((17 << 48) | (19 << 32) | (23 << 16) | 29),
                ],
                shape: vec![1, 4],
            },
            Tensor {
                data: vec![
                    -((31 << 16) | 37),
                    (41 << 32) | (43 << 16) | 13,
                    -((53 << 48) | (59 << 32) | (61 << 16) | 67),
                    1,
                ],
                shape: vec![1, 4],
            },
            Tensor {
                data: vec![
                    (73 << 32) | (79 << 16) | 83,
                    -((89 << 48) | (97 << 32) | (2 << 16) | 3),
                    5,
                    -((7 << 16) | 11),
                ],
                shape: vec![1, 4],
            },
            Tensor {
                data: vec![
                    -((13 << 48) | (17 << 32) | (19 << 16) | 23),
                    29,
                    -((31 << 16) | 37),
                    (41 << 32) | (43 << 16) | 53,
                ],
                shape: vec![1, 4],
            },
        ]);

        let truncate_bits = if should_rescale { i64::PRECISION } else { 0 };
        let result = Tensors(
            inputs
                .0
                .iter()
                .map(|input| Tensor {
                    data: input
                        .data
                        .iter()
                        .map(|x| (*x).max(0).divided_by(truncate_bits))
                        .collect_vec(),
                    shape: inputs.0[0].shape.clone(),
                })
                .collect_vec(),
        );

        let relu = Relu::<i64, Hyrax>::new(should_rescale);
        let (outputs, relu_witness, lookup_witness) = relu.forward(&inputs);
        let relu_witness: ReluWitness<_> = relu_witness.try_into().unwrap();

        assert_eq!(outputs, result);

        let expected_wit = ReluWitness {
            is_positive: DenseMultilinearExtension::from_evaluations_vec(
                4,
                field_vec![Fr, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
            ),
            in_decomp: vec![
                DenseMultilinearExtension::from_evaluations_vec(
                    4,
                    field_vec![Fr, 1, 37, 83, 23, 5, 13, 3, 29, 13, 67, 5, 37, 29, 1, 11, 53],
                ),
                DenseMultilinearExtension::from_evaluations_vec(
                    4,
                    field_vec![Fr, 0, 31, 79, 19, 3, 43, 2, 0, 11, 61, 0, 31, 23, 0, 7, 43],
                ),
                DenseMultilinearExtension::from_evaluations_vec(
                    4,
                    field_vec![Fr, 0, 0, 73, 17, 0, 41, 97, 0, 7, 59, 0, 0, 19, 0, 0, 41],
                ),
                DenseMultilinearExtension::from_evaluations_vec(
                    4,
                    field_vec![Fr, 0, 0, 0, 13, 0, 0, 89, 0, 0, 53, 0, 0, 17, 0, 0, 0],
                ),
            ],
        };
        assert_eq!(relu_witness.is_positive, expected_wit.is_positive);
        assert_eq!(relu_witness.in_decomp, expected_wit.in_decomp);

        let expected_lookup_wit = LookupWitness {
            range: field_vec![
                Fr, 1, 37, 83, 23, 5, 13, 3, 29, 13, 67, 5, 37, 29, 1, 11, 53, 0, 31, 79, 19, 3,
                43, 2, 0, 11, 61, 0, 31, 23, 0, 7, 43, 0, 0, 73, 17, 0, 41, 97, 0, 7, 59, 0, 0, 19,
                0, 0, 41, 0, 0, 0, 13, 0, 0, 89, 0, 0, 53, 0, 0, 17, 0, 0, 0,
            ],
        };

        assert_eq!(lookup_witness.range, expected_lookup_wit.range);
        let inputs = inputs.into_poly();
        let outputs = outputs.into_poly();
        ReluInstance {
            relu,
            inputs,
            outputs,

            relu_witness,
            lookup_witness,

            instance_num_vars: 2,
        }
    }

    #[test]
    fn test_forward_rescale() {
        test_helper(true);
    }

    #[test]
    fn test_forward_not_rescale() {
        test_helper(false);
    }

    fn prove_and_verify_i64(should_rescale: bool) {
        let mut rng = rand::thread_rng();

        let ReluInstance {
            relu,
            inputs,
            outputs,

            relu_witness,
            lookup_witness,

            instance_num_vars,
        } = test_helper(should_rescale);

        let out_point = (0..outputs.num_vars())
            .map(|_| Fr::random(&mut rng))
            .collect_vec();
        let (lo_point, hi_point) = out_point.split_at(instance_num_vars);
        let lookup_point = (0..lookup_witness.range.len().log_2())
            .map(|_| Fr::random(&mut rng))
            .collect_vec();
        let out_evaluation = outputs.evaluate(&out_point);
        let lookup_witness_vec = lookup_witness
            .range
            .iter()
            .map(|x| Fr::from(*x))
            .collect_vec();
        let lookup_evaluation = DenseMultilinearExtension::from_evaluations_slice(
            lookup_witness_vec.len().log_2(),
            &lookup_witness_vec,
        )
        .evaluate(&lookup_point);

        let (prover, verifier) = relu.prover_and_verifier();
        let proof = {
            let mut transcript = ProofTranscript::new(b"test");
            let LayerProverOutput { proof, .. } = prover
                .prove(
                    inputs.clone(),
                    relu_witness.into(),
                    lo_point,
                    hi_point,
                    &lookup_point,
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
            let lookup_claim = LookupClaim {
                point: lookup_point,
                range: lookup_evaluation,
            };
            verifier
                .verify(&output_claim, &Some(lookup_claim), proof, &mut transcript)
                .expect("Verifier failed")
        };
    }

    #[test]
    fn test_prove_and_verify_rescale() {
        prove_and_verify_i64(true);
    }

    #[test]
    fn test_prove_and_verify_not_rescale() {
        prove_and_verify_i64(false);
    }
}
