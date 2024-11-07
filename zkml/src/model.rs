use ark_std::{end_timer, start_timer};
use gkr::logup::{
    logup_prover, logup_verifier, logup_witness_gen_multi_input, LogupInputs, LogupProof,
    LogupProverOutput, LogupWitness,
};
use itertools::{izip, Itertools};
use pcs::{
    commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme},
    field::JoltField,
    poly::dense::DenseMultilinearExtension,
    utils::transcript::{AppendToTranscript, ProofTranscript},
};

use crate::{
    error::ZKMLError,
    layers::{
        LayerGenerator, LayerProof, LayerProver, LayerProverOutput, LayerVerifier, LayerWitness,
    },
    lookup::{compute_logup_inputs, compute_lookup_claims, LookupWitness},
    tensor::{TensorClaim, TensorData, Tensors},
};

pub struct Model<T: TensorData, PCS: CommitmentScheme> {
    pub nodes: Vec<Box<dyn LayerGenerator<T, PCS>>>,
}

impl<T: TensorData + Into<PCS::Field>, PCS: CommitmentScheme> Model<T, PCS> {
    pub fn prover_and_verifier(self) -> (ModelProver<PCS>, ModelVerifier<PCS>) {
        let mut provers: Vec<Box<dyn LayerProver<PCS>>> = vec![];
        let mut verifiers: Vec<Box<dyn LayerVerifier<PCS>>> = vec![];
        for node in self.nodes.into_iter() {
            let (prover, verifier) = (*node).prover_and_verifier();
            provers.push(prover);
            verifiers.push(verifier);
        }

        (
            ModelProver {
                layers: provers,
                range_bit_width: T::RANGE_BIT_WIDTH,
            },
            ModelVerifier { layers: verifiers },
        )
    }

    #[allow(clippy::type_complexity)]
    pub fn forward(
        &self,
        data: Tensors<T>,
    ) -> (
        Vec<DenseMultilinearExtension<PCS::Field>>,
        Vec<LayerWitness<PCS::Field>>,
        Vec<LookupWitness<PCS::Field>>,
    ) {
        let nodes = &self.nodes;
        let mut inputs = Vec::with_capacity(nodes.len());
        inputs.push(data);
        let mut wits = Vec::with_capacity(nodes.len());
        let mut lookups = Vec::with_capacity(nodes.len());
        // Currently we only support sequential layer structure: L1 -> L2 -> L3 -> ...
        for node in nodes.iter() {
            let input = inputs.last().unwrap();
            let (output, wit, lookup) = node.forward(input);
            wits.push(wit);
            lookups.push(lookup);
            inputs.push(output);
        }
        // Convert from Tensor<T> to DenseMultilinearExtension<F>.
        let inputs = inputs
            .into_iter()
            .map(|inputs| inputs.into_poly())
            .collect_vec();
        (inputs, wits, lookups)
    }
}

pub struct ModelCommitment<PCS: CommitmentScheme> {
    pub layers: Vec<Vec<PCS::Commitment>>,
    pub range: Vec<PCS::Commitment>,
}

pub struct ModelProver<PCS: CommitmentScheme> {
    pub layers: Vec<Box<dyn LayerProver<PCS>>>,
    pub range_bit_width: usize,
}

pub struct ModelProof<PCS>
where
    PCS: CommitmentScheme,
{
    pub range_proof: LogupProof<PCS::Field>,
    pub layer_proofs: Vec<LayerProof<PCS>>,
    pub layer_openings: Vec<Option<PCS::BatchedProof>>,
    pub range_table_opening: PCS::BatchedProof,
}

pub struct ModelVerifier<PCS: CommitmentScheme> {
    pub layers: Vec<Box<dyn LayerVerifier<PCS>>>,
}

impl<PCS: CommitmentScheme> ModelProver<PCS> {
    #[allow(clippy::type_complexity)]
    pub fn commit_phase(
        &self,
        wits: &[LayerWitness<PCS::Field>],
        lookups: Vec<LookupWitness<PCS::Field>>,
        transcript: &mut ProofTranscript,
    ) -> Result<
        (
            LogupInputs<PCS::Field>,
            LogupWitness<PCS::Field>,
            ModelCommitment<PCS>,
            Vec<Option<PCS::Setup>>,
            PCS::Setup,
        ),
        ZKMLError,
    > {
        // Generate commitment of nodes.
        let (layers, srs) = wits
            .iter()
            .map(|wit| {
                if wit.0.is_empty() {
                    return (vec![], None);
                }
                let shapes = vec![CommitShape::new(wit.0[0].len(), BatchType::Small); wit.0.len()];
                let srs = PCS::setup(&shapes);
                let comms = PCS::batch_commit_polys(&wit.0, &srs, BatchType::Small);
                for comm in comms.iter() {
                    comm.append_to_transcript(transcript);
                }
                (comms, Some(srs))
            })
            .unzip();

        // Generate the range table [0, 1, ..., 2^range_bit_width - 1].
        let range_inputs = {
            let table_items = (0..(1 << self.range_bit_width))
                .map(|x| PCS::Field::from_u64(x).unwrap())
                .collect_vec();
            compute_logup_inputs(lookups, table_items)
        }?;

        // Commit table items and count.
        let range = &[
            range_inputs.table_items.as_slice(),
            range_inputs.count.as_slice(),
        ];
        let range_srs = PCS::setup(&vec![
            CommitShape::new(
                1 << self.range_bit_width,
                BatchType::Small
            );
            2
        ]);
        let range = PCS::batch_commit(range, &range_srs, BatchType::Small);
        range
            .iter()
            .for_each(|comm| comm.append_to_transcript(transcript));

        let lookup_challenge = transcript.challenge_scalar();
        let range_witness = logup_witness_gen_multi_input(&range_inputs, lookup_challenge)?;

        Ok((
            range_inputs,
            range_witness,
            ModelCommitment { range, layers },
            srs,
            range_srs,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prove_phase(
        &self,
        model_srs: &[Option<PCS::Setup>],
        range_srs: &PCS::Setup,

        logup_inputs: LogupInputs<PCS::Field>,
        logup_witness: LogupWitness<PCS::Field>,

        mut inputs: Vec<DenseMultilinearExtension<PCS::Field>>,
        wits: Vec<LayerWitness<PCS::Field>>,

        transcript: &mut ProofTranscript,
        instance_num_vars: usize,
    ) -> Result<ModelProof<PCS>, ZKMLError> {
        let layers = &self.layers;

        // Generate logup proof
        let timer = start_timer!(|| "logup_prover");
        let LogupProverOutput {
            proof: range_proof,
            input_points,
            table_point: range_point,
        } = logup_prover(&logup_witness, transcript)?;
        end_timer!(timer);

        // Generate logup table opening
        let timer = start_timer!(|| "logup_table_opening");
        let range_table_opening = {
            let LogupInputs {
                table_items, count, ..
            } = logup_inputs;
            let polys = [&table_items, &count]
                .into_iter()
                .map(|v| DenseMultilinearExtension::from_evaluations_slice(self.range_bit_width, v))
                .collect_vec();
            let poly_refs = polys.iter().collect_vec();
            let evals = polys
                .iter()
                .map(|poly| poly.evaluate(&range_point))
                .collect_vec();
            PCS::batch_prove(
                range_srs,
                &poly_refs,
                &range_point,
                &evals,
                BatchType::Small,
                transcript,
            )
        };
        end_timer!(timer);

        // Generate node proofs
        let mut layer_proofs = Vec::with_capacity(layers.len());
        let mut layer_openings = Vec::with_capacity(layers.len());
        let output = inputs.pop().unwrap();
        let mut out_point = transcript.challenge_vector(output.num_vars());

        for (layer, srs, input, wit, lookup_point) in izip!(
            layers.iter(),
            model_srs.iter(),
            inputs.into_iter(),
            wits.into_iter(),
            input_points.into_iter()
        )
        .rev()
        {
            let (lo_point, hi_point) = out_point.split_at(instance_num_vars);
            let LayerProverOutput {
                proof,
                lo_point,
                hi_point,
            } = layer.prove(
                input,
                wit.clone(),
                lo_point,
                hi_point,
                &lookup_point,
                transcript,
            )?;

            let timer = start_timer!(|| "layer_opening");
            let polys = wit.0.iter().collect_vec();
            out_point = [lo_point, hi_point].concat();
            let opening = if let Some(srs) = srs {
                Some(PCS::batch_prove(
                    srs,
                    &polys,
                    &out_point,
                    &proof.claims[..polys.len()],
                    BatchType::Small,
                    transcript,
                ))
            } else {
                None
            };
            end_timer!(timer);
            layer_proofs.push(proof);
            layer_openings.push(opening);
        }
        layer_proofs.reverse();
        layer_openings.reverse();

        Ok(ModelProof {
            range_table_opening,
            range_proof,
            layer_proofs,
            layer_openings,
        })
    }
}

impl<PCS: CommitmentScheme> ModelVerifier<PCS> {
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        model_comm: ModelCommitment<PCS>,
        model_srs: &[Option<PCS::Setup>],

        range_srs: PCS::Setup,
        proof: ModelProof<PCS>,
        output: &DenseMultilinearExtension<PCS::Field>,

        transcript: &mut ProofTranscript,
        instance_num_vars: usize,
    ) -> Result<(), ZKMLError> {
        let ModelProof {
            range_table_opening,
            range_proof,
            layer_proofs,
            layer_openings,
        } = proof;

        // Receive commitments
        let ModelCommitment {
            range: range_comms,
            layers: layer_comms,
        } = model_comm;
        for comm in layer_comms.iter().flatten().chain(range_comms.iter()) {
            comm.append_to_transcript(transcript);
        }

        let lookup_challenge = transcript.challenge_scalar();

        // Verify logup proof
        let (range_claims, range_table_claims) = {
            let logup_claim = logup_verifier(range_proof, transcript)?;
            compute_lookup_claims(logup_claim, lookup_challenge)
        };

        // Verify table opening
        let range_table_comms_ref = range_comms.iter().collect_vec();
        PCS::batch_verify(
            &range_table_opening,
            &range_srs,
            &range_table_claims.point,
            &[range_table_claims.range, range_table_claims.count],
            &range_table_comms_ref,
            transcript,
        )?;

        // Verify node proofs
        let out_point = transcript.challenge_vector(output.num_vars());
        let value = output.evaluate(&out_point);
        let (lo_point, hi_point) = out_point.split_at(instance_num_vars);
        let mut output_claim = TensorClaim {
            lo_point: lo_point.to_vec(),
            hi_point: hi_point.to_vec(),
            value,
        };
        for (layer, srs, poly_comms, proof, opening, range_claim) in izip!(
            self.layers.iter(),
            model_srs.iter(),
            layer_comms.iter(),
            layer_proofs.into_iter(),
            layer_openings.into_iter(),
            range_claims.iter()
        )
        .rev()
        {
            let (input_claim, layer_claims) =
                layer.verify(&output_claim, range_claim, proof, transcript)?;

            let comm_refs = poly_comms.iter().collect_vec();
            let claims = layer_claims.iter().map(|claim| claim.value).collect_vec();
            if let (Some(srs), Some(opening)) = (srs, opening) {
                let point = [input_claim.lo_point.clone(), input_claim.hi_point.clone()].concat();
                PCS::batch_verify(&opening, srs, &point, &claims, &comm_refs, transcript)?;
            }
            output_claim = input_claim;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::G1Projective;
    use ark_std::test_rng;
    use itertools::Itertools;
    use pcs::{
        commitment::{commitment_scheme::CommitmentScheme, hyrax::HyraxScheme},
        poly::ternary_sparse::TernarySparseMultilinearExtension,
        utils::{math::Math, transcript::ProofTranscript},
    };
    use rand::{seq::SliceRandom, Rng};

    use crate::{
        layers::{
            linear::Linear, relu::Relu, sparse_linear::SparseLinear,
            ternary_sparse_linear::TernarySparseLinear,
        },
        tensor::{SparseTensor, Tensor, TensorData, Tensors, TernarySparseTensor},
    };

    use super::Model;

    type Hyrax = HyraxScheme<G1Projective>;

    fn construct_linear<T: TensorData + Into<PCS::Field>, PCS: CommitmentScheme>(
        in_size: usize,
        out_size: usize,
    ) -> Box<Linear<T, PCS>> {
        let rng = &mut test_rng();
        let padded_in_size = in_size.next_power_of_two();
        let padded_out_size = out_size.next_power_of_two();

        let transposed_weight = {
            let mut data = (0..in_size)
                .map(|_| {
                    let mut data = (0..out_size).map(|_| T::random(rng)).collect_vec();
                    data.resize(padded_out_size, T::zero());
                    data
                })
                .collect_vec();
            data.extend(vec![
                vec![T::zero(); padded_out_size];
                padded_in_size - in_size
            ]);
            let data = data.into_iter().flatten().collect_vec();
            Tensor {
                data,
                shape: vec![padded_in_size, padded_out_size],
            }
        };
        let scaled_bias = {
            let mut data = (0..out_size).map(|_| T::random(rng)).collect_vec();
            data.resize(padded_out_size, T::zero());
            Tensor {
                data,
                shape: vec![1, padded_out_size],
            }
        };
        Box::new(Linear::<_, PCS>::new(transposed_weight, scaled_bias, true))
    }

    fn construct_sparse_linear<T: TensorData + Into<PCS::Field>, PCS: CommitmentScheme>(
        in_size: usize,
        out_size: usize,
    ) -> Box<SparseLinear<T, PCS>> {
        let rng = &mut test_rng();
        let padded_in_size = in_size.next_power_of_two();
        let padded_out_size = out_size.next_power_of_two();

        let num_vars = (padded_in_size * padded_out_size).log_2();
        let num_nonzero = padded_in_size * padded_out_size / 5; // Randomly set one.

        let transposed_weight = {
            let mut range: Vec<usize> = (0..(1 << num_vars)).collect_vec();
            range.shuffle(rng);

            let data = (0..num_nonzero)
                .map(|i| (range[i], T::random(rng)))
                .collect_vec();
            SparseTensor {
                data,
                shape: vec![padded_in_size, padded_out_size],
            }
        };
        let scaled_bias = {
            let mut data = (0..out_size).map(|_| T::random(rng)).collect_vec();
            data.resize(padded_out_size, T::zero());
            Tensor {
                data,
                shape: vec![1, padded_out_size],
            }
        };
        Box::new(SparseLinear::<_, PCS>::new(
            transposed_weight,
            scaled_bias,
            true,
        ))
    }

    fn construct_ternary_sparse_linear<T: TensorData + Into<PCS::Field>, PCS: CommitmentScheme>(
        in_size: usize,
        out_size: usize,
    ) -> Box<TernarySparseLinear<T, PCS>> {
        let rng = &mut test_rng();
        let padded_in_size = in_size.next_power_of_two();
        let padded_out_size = out_size.next_power_of_two();

        let num_vars = (padded_in_size * padded_out_size).log_2();
        let num_nonzero = padded_in_size * padded_out_size / 5; // Randomly set one.

        let transposed_weight = {
            let tmp = TernarySparseMultilinearExtension::<PCS::Field>::rand_with_config(
                num_vars,
                num_nonzero,
                false,
                rng,
            );
            TernarySparseTensor {
                pos: tmp.evaluations_pos,
                neg: tmp.evaluations_neg,
                shape: vec![padded_in_size, padded_out_size],
            }
        };
        let scaled_bias = {
            let mut data = (0..out_size).map(|_| T::random(rng)).collect_vec();
            data.resize(padded_out_size, T::zero());
            Tensor {
                data,
                shape: vec![1, padded_out_size],
            }
        };
        Box::new(TernarySparseLinear::<_, PCS>::new(
            transposed_weight,
            scaled_bias,
            true,
        ))
    }

    fn construct_relu<T: TensorData + Into<PCS::Field>, PCS: CommitmentScheme>(
        should_rescale: bool,
    ) -> Box<Relu<T, PCS>> {
        Box::new(Relu::new(should_rescale))
    }

    fn construct_linear_model<
        T: TensorData + Into<PCS::Field> + From<bool> + 'static,
        PCS: CommitmentScheme,
    >() -> Model<T, PCS> {
        // Construct a simple neural network model:
        //  linear(4, 20) -> relu -> linear(20, 20) -> relu -> linear(20, 3) -> relu
        Model {
            nodes: vec![
                construct_linear::<T, PCS>(4, 20),
                construct_relu::<T, PCS>(true),
                construct_linear::<T, PCS>(20, 20),
                construct_relu::<T, PCS>(true),
                construct_linear::<T, PCS>(20, 3),
                construct_relu::<T, PCS>(true),
            ],
        }
    }

    fn construct_sparse_linear_model<
        T: TensorData + Into<PCS::Field> + From<bool> + 'static,
        PCS: CommitmentScheme,
    >() -> Model<T, PCS> {
        // Construct a simple neural network model:
        //  linear(4, 20) -> relu -> linear(20, 20) -> relu -> linear(20, 3) -> relu
        Model {
            nodes: vec![
                construct_sparse_linear::<T, PCS>(4, 20),
                construct_relu::<T, PCS>(true),
                construct_sparse_linear::<T, PCS>(20, 20),
                construct_relu::<T, PCS>(true),
                construct_sparse_linear::<T, PCS>(20, 3),
                construct_relu::<T, PCS>(true),
            ],
        }
    }

    fn construct_ternary_sparse_linear_model<
        T: TensorData + Into<PCS::Field> + From<bool> + 'static,
        PCS: CommitmentScheme,
    >() -> Model<T, PCS> {
        // Construct a simple neural network model:
        //  linear(4, 20) -> relu -> linear(20, 20) -> relu -> linear(20, 3) -> relu
        Model {
            nodes: vec![
                construct_ternary_sparse_linear::<T, PCS>(4, 20),
                construct_relu::<T, PCS>(false),
                construct_ternary_sparse_linear::<T, PCS>(20, 20),
                construct_relu::<T, PCS>(false),
                construct_ternary_sparse_linear::<T, PCS>(20, 3),
                construct_relu::<T, PCS>(false),
            ],
        }
    }

    fn test_e2e(model: impl Fn() -> Model<i64, Hyrax>) {
        let instance_num_vars = 3;
        let model = model();
        // Witness generation.
        let rng = &mut test_rng();
        let data = Tensors(
            (0..1 << instance_num_vars)
                .map(|_| Tensor {
                    data: (0..4).map(|_| rng.gen::<i32>() as i64).collect_vec(),
                    shape: vec![1, 4],
                })
                .collect_vec(),
        );

        let (inputs, wits, lookups) = model.forward(data);
        let outputs = inputs.last().unwrap().clone();

        // Prove
        let (prover, verifier) = model.prover_and_verifier();
        let mut transcript = ProofTranscript::new(b"zkml");
        let (logup_inputs, logup_witness, comm, srs, table_srs) = prover
            .commit_phase(&wits, lookups, &mut transcript)
            .unwrap();
        let proof = prover
            .prove_phase(
                &srs,
                &table_srs,
                logup_inputs,
                logup_witness,
                inputs,
                wits,
                &mut transcript,
                instance_num_vars,
            )
            .unwrap();

        // Verifier
        let mut transcript = ProofTranscript::new(b"zkml");
        verifier
            .verify(
                comm,
                &srs,
                table_srs,
                proof,
                &outputs,
                &mut transcript,
                instance_num_vars,
            )
            .expect("verification failed");
    }

    #[test]
    fn test_e2e_linear() {
        test_e2e(construct_linear_model);
    }

    #[test]
    fn test_e2e_sparse_linear() {
        test_e2e(construct_sparse_linear_model);
    }

    #[test]
    fn test_e2e_ternary_sparse_linear() {
        test_e2e(construct_ternary_sparse_linear_model)
    }
}
