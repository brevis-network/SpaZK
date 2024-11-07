use ark_bn254::G1Projective;
use ark_std::{end_timer, start_timer, test_rng};
use itertools::Itertools;
use pcs::{
    commitment::{commitment_scheme::CommitmentScheme, hyrax::HyraxScheme},
    poly::ternary_sparse::TernarySparseMultilinearExtension,
    utils::{math::Math, transcript::ProofTranscript},
};
use rand::{seq::SliceRandom, Rng};
use zkml::{
    layers::{
        linear::Linear, relu::Relu, sparse_linear::SparseLinear,
        ternary_sparse_linear::TernarySparseLinear,
    },
    model::Model,
    tensor::{SparseTensor, Tensor, TensorData, Tensors, TernarySparseTensor},
};

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
    sparsity: usize,
) -> Box<TernarySparseLinear<T, PCS>> {
    let rng = &mut test_rng();
    let padded_in_size = in_size.next_power_of_two();
    let padded_out_size = out_size.next_power_of_two();

    let num_vars = (padded_in_size * padded_out_size).log_2();
    let num_nonzero = (padded_in_size * padded_out_size) >> sparsity; // Randomly set one.

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
>(
    _sparsity: usize,
) -> Model<T, PCS> {
    // Construct a simple neural network model:
    Model {
        nodes: vec![
            construct_linear::<T, PCS>(784, 1024),
            construct_relu::<T, PCS>(true),
            construct_linear::<T, PCS>(1024, 10),
            construct_relu::<T, PCS>(true),
        ],
    }
}

#[allow(unused)]
fn construct_sparse_linear_model<
    T: TensorData + Into<PCS::Field> + From<bool> + 'static,
    PCS: CommitmentScheme,
>() -> Model<T, PCS> {
    // Construct a simple neural network model:
    Model {
        nodes: vec![
            construct_linear::<T, PCS>(784, 1024),
            construct_relu::<T, PCS>(true),
            construct_linear::<T, PCS>(1024, 10),
            construct_relu::<T, PCS>(true),
        ],
    }
}

fn construct_ternary_sparse_linear_model<
    T: TensorData + Into<PCS::Field> + From<bool> + 'static,
    PCS: CommitmentScheme,
>(
    sparsity: usize,
) -> Model<T, PCS> {
    // Construct a simple neural network model:
    Model {
        nodes: vec![
            construct_ternary_sparse_linear::<T, PCS>(784, 1024, sparsity),
            construct_relu::<T, PCS>(false),
            construct_ternary_sparse_linear::<T, PCS>(1024, 10, sparsity),
            construct_relu::<T, PCS>(false),
        ],
    }
}

fn run_model<T: TensorData + Into<PCS::Field>, PCS: CommitmentScheme>(
    model: impl Fn(usize) -> Model<T, PCS>,
    #[allow(unused)] name: String,
    instance_num_vars: usize,
    sparsity: usize,
    data: Tensors<T>,
) {
    let model = model(sparsity);
    // Witness generation.
    let timer = start_timer!(|| format!("{}, log(ins) = {}, forward", name, instance_num_vars));
    let (inputs, wits, lookups) = model.forward(data);
    let outputs = inputs.last().unwrap().clone();
    end_timer!(timer);
    let timer =
        start_timer!(|| format!("{}, log(ins) = {}, commit_model", name, instance_num_vars));
    let (prover, verifier) = model.prover_and_verifier();
    end_timer!(timer);

    // Prove
    let timer =
        start_timer!(|| format!("{}, log(ins) = {}, commit_phase", name, instance_num_vars));
    let mut transcript = ProofTranscript::new(b"zkml");
    let (logup_inputs, logup_witness, comm, srs, table_srs) = prover
        .commit_phase(&wits, lookups, &mut transcript)
        .unwrap();
    end_timer!(timer);
    let timer = start_timer!(|| format!("{}, log(ins) = {}, prove_phase", name, instance_num_vars));
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
    end_timer!(timer);

    // Verifier
    let timer = start_timer!(|| format!("{}, log(ins) = {}, verify", name, instance_num_vars));
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
    end_timer!(timer);
}

fn main() {
    let instance_num_vars = 4;
    let rng = &mut test_rng();
    let data = Tensors(
        (0..1 << instance_num_vars)
            .map(|_| Tensor {
                data: [
                    (0..28)
                        .flat_map(|_| {
                            [
                                (0..28).map(|_| rng.gen::<i32>() as i64).collect_vec(),
                                vec![0; 4],
                            ]
                            .concat()
                        })
                        .collect_vec(),
                    vec![0; 32 * 4],
                ]
                .concat(),
                shape: vec![1, 1024],
            })
            .collect_vec(),
    );
    assert_eq!(data.0[0].data.len(), 1024);
    run_model(
        construct_linear_model::<i64, Hyrax>,
        "linear".to_string(),
        instance_num_vars,
        0,
        data.clone(),
    );

    for sparsity in (0..5).rev() {
        run_model(
            construct_ternary_sparse_linear_model::<i64, Hyrax>,
            format!("ternary-1/{}", (1 << sparsity)),
            instance_num_vars,
            sparsity,
            data.clone(),
        );
    }
}
