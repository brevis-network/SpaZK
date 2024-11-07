use ark_std::{end_timer, start_timer};
use itertools::{izip, Itertools};
use pcs::{
    field::JoltField,
    poly::dense::DenseMultilinearExtension,
    utils::{precompute_eq, transcript::ProofTranscript},
};
use rand::{rngs::OsRng, Rng};
use std::rc::Rc;
use sumcheck::{
    error::Error,
    protocol::{
        prover::ProverMsg, verifier::SubClaim, ListOfProductsOfPolynomials, PolynomialInfo,
    },
    MLSumcheck,
};

fn eq_eval<F: JoltField>(x: &[F], y: &[F]) -> F {
    assert_eq!(x.len(), y.len());
    let one = F::one();
    izip!(x, y)
        .map(|(x, y)| (*x * y) + (one - x) * (one - y))
        .product()
}

fn logup_witness_gen<F: JoltField>(
    input_num_vars: usize,
    table_num_vars: usize,
) -> (Vec<Vec<Vec<F>>>, Vec<Vec<Vec<F>>>) {
    let mut csprng = OsRng;
    let table_size = 1 << table_num_vars;
    let table = (0..table_size)
        .map(|_| F::random(&mut csprng))
        .collect_vec();
    let mut count = vec![0; table_size];

    let input_size = 1 << input_num_vars;
    let input = (0..input_size)
        .map(|_| {
            let index = csprng.gen_range(0..table_size);
            count[index] += 1;
            table[index]
        })
        .collect_vec();

    let count = count
        .into_iter()
        .map(|x| F::from_u64(x).unwrap())
        .collect_vec();

    let compute_frac_tree = |tree_cells: &mut Vec<Vec<Vec<F>>>, num_vars: usize| {
        for i in (0..num_vars).rev() {
            let size = 1 << i;
            let d = (0..size)
                .map(|j| tree_cells[i + 1][0][j] * tree_cells[i + 1][0][j + size])
                .collect_vec();
            tree_cells[i][0] = d;
            let n = (0..size)
                .map(|j| {
                    tree_cells[i + 1][0][j] * tree_cells[i + 1][1][j + size]
                        + tree_cells[i + 1][0][j + size] * tree_cells[i + 1][1][j]
                })
                .collect_vec();
            tree_cells[i][1] = n;
        }
    };

    let table_tree = {
        let mut table_tree = vec![vec![vec![], vec![]]; table_num_vars + 1];
        table_tree[table_num_vars] = vec![table, count];
        compute_frac_tree(&mut table_tree, table_num_vars);
        table_tree
    };

    let input_tree = {
        let mut input_tree = vec![vec![vec![], vec![]]; input_num_vars + 1];
        input_tree[input_num_vars] = vec![input, vec![F::one(); input_size]];
        compute_frac_tree(&mut input_tree, input_num_vars);
        input_tree
    };

    (input_tree, table_tree)
}

struct FracSumProof<F: JoltField> {
    sum: [F; 2],
    sumcheck_proofs: Vec<(Vec<ProverMsg<F>>, [F; 4])>,
}

struct LogupProof<F: JoltField> {
    input_proof: FracSumProof<F>,
    table_proof: FracSumProof<F>,
}

fn logup_prover<F: JoltField>(
    input_tree: Vec<Vec<Vec<F>>>,
    table_tree: Vec<Vec<Vec<F>>>,
    transcript: &mut ProofTranscript,
    input_num_vars: usize,
    table_num_vars: usize,
) -> Result<LogupProof<F>, Error> {
    let timer = start_timer!(|| format!(
        "input_num_vars: {}, table_num_vars: {}, logup sumcheck-prove",
        input_num_vars, table_num_vars
    ));
    let input_sum = [input_tree[0][0][0], input_tree[0][1][0]];
    let input_proof = frac_sum_prover(input_sum, &input_tree, transcript)?;
    let table_sum = [table_tree[0][0][0], table_tree[0][1][0]];
    let table_proof = frac_sum_prover(table_sum, &table_tree, transcript)?;
    end_timer!(timer);
    Ok(LogupProof {
        input_proof,
        table_proof,
    })
}

struct FracSumClaim<F: JoltField> {
    point: Vec<F>,
    /// [Denominator evaluation, numerator evaluation]
    evaluations: [F; 2],
}

struct LogupClaim<F: JoltField> {
    input_claim: FracSumClaim<F>,
    table_claim: FracSumClaim<F>,
}

fn logup_verifier<F: JoltField>(
    proof: LogupProof<F>,
    transcript: &mut ProofTranscript,
    input_num_vars: usize,
    table_num_vars: usize,
) -> Result<LogupClaim<F>, Error> {
    let timer = start_timer!(|| format!(
        "input_num_vars: {}, table_num_vars: {}, logup sumcheck-verify",
        input_num_vars, table_num_vars
    ));
    let input_claim = frac_sum_verifier(proof.input_proof, transcript)?;
    let table_claim = frac_sum_verifier(proof.table_proof, transcript)?;
    end_timer!(timer);
    Ok(LogupClaim {
        input_claim,
        table_claim,
    })
}

fn frac_sum_prover<F: JoltField>(
    sum: [F; 2],
    tree_cells: &[Vec<Vec<F>>],
    transcript: &mut ProofTranscript,
) -> Result<FracSumProof<F>, Error> {
    transcript.append_scalar(&sum[0]);
    transcript.append_scalar(&sum[1]);
    let mut point = vec![];
    let mut sumcheck_proofs = vec![];
    for depth in 0..tree_cells.len() - 1 {
        let alpha = transcript.challenge_scalar();
        if depth == 0 {
            transcript.append_scalar(&tree_cells[1][0][0]);
            transcript.append_scalar(&tree_cells[1][0][1]);
            transcript.append_scalar(&tree_cells[1][1][0]);
            transcript.append_scalar(&tree_cells[1][1][1]);
            let r = transcript.challenge_scalar();
            point = vec![r];
            let evals = [
                tree_cells[1][0][0],
                tree_cells[1][0][1],
                tree_cells[1][1][0],
                tree_cells[1][1][1],
            ];
            sumcheck_proofs.push((vec![], evals));
            continue;
        }

        let (d_left, d_right) = tree_cells[depth + 1][0].split_at(1 << depth);
        let d_left = Rc::new(DenseMultilinearExtension::from_evaluations_slice(
            depth, d_left,
        ));
        let d_right = Rc::new(DenseMultilinearExtension::from_evaluations_slice(
            depth, d_right,
        ));

        let (n_left, n_right) = tree_cells[depth + 1][1].split_at(1 << depth);
        let n_left = Rc::new(DenseMultilinearExtension::from_evaluations_slice(
            depth, n_left,
        ));
        let n_right = Rc::new(DenseMultilinearExtension::from_evaluations_slice(
            depth, n_right,
        ));

        let eq = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            depth,
            precompute_eq(&point),
        ));

        let mut sc_poly = ListOfProductsOfPolynomials::new(depth);
        sc_poly.add_product([eq.clone(), d_left.clone(), d_right.clone()], F::one());
        sc_poly.add_product([eq.clone(), d_left, n_right], alpha);
        sc_poly.add_product([eq, n_left, d_right], alpha);
        let (proof, state) =
            MLSumcheck::prove_as_subprotocol(transcript, &sc_poly).expect("sumcheck proof failed");

        let sumcheck::protocol::prover::ProverState {
            randomness,
            flattened_ml_extensions,
            ..
        } = state;

        let last_randomness = randomness.last().unwrap();
        let d_left_eval = flattened_ml_extensions[1].fix_variables(&[*last_randomness])[0];
        let d_right_eval = flattened_ml_extensions[2].fix_variables(&[*last_randomness])[0];
        let n_right_eval = flattened_ml_extensions[3].fix_variables(&[*last_randomness])[0];
        let n_left_eval = flattened_ml_extensions[4].fix_variables(&[*last_randomness])[0];
        transcript.append_scalar(&d_left_eval);
        transcript.append_scalar(&d_right_eval);
        transcript.append_scalar(&n_left_eval);
        transcript.append_scalar(&n_right_eval);

        let r = transcript.challenge_scalar();
        point = randomness;
        point.push(r);

        sumcheck_proofs.push((
            proof,
            [d_left_eval, d_right_eval, n_left_eval, n_right_eval],
        ));
    }
    Ok(FracSumProof {
        sum,
        sumcheck_proofs,
    })
}

fn frac_sum_verifier<F: JoltField>(
    proof: FracSumProof<F>,
    transcript: &mut ProofTranscript,
) -> Result<FracSumClaim<F>, Error> {
    let FracSumProof {
        sum,
        sumcheck_proofs,
    } = proof;
    transcript.append_scalar(&sum[0]);
    transcript.append_scalar(&sum[1]);
    sumcheck_proofs.into_iter().enumerate().try_fold(
        FracSumClaim {
            point: vec![],
            evaluations: sum,
        },
        |last_claim, (depth, sumcheck_proof)| {
            let FracSumClaim {
                point: last_point,
                evaluations: [d_eval, n_eval],
            } = last_claim;
            let (proof, evals) = sumcheck_proof;

            let alpha: F = transcript.challenge_scalar();
            let (mut point, expected_evaluation) = if last_point.is_empty() {
                (vec![], d_eval + n_eval * alpha)
            } else {
                let poly_info = PolynomialInfo {
                    max_multiplicands: 3,
                    num_variables: last_point.len(),
                };
                let SubClaim {
                    point,
                    expected_evaluation,
                } = MLSumcheck::verify_as_subprotocol(
                    transcript,
                    &poly_info,
                    d_eval + n_eval * alpha,
                    &proof,
                )?;
                (point, expected_evaluation)
            };

            let eq_eval = eq_eval(&last_point, &point);
            let [d_left_eval, d_right_eval, n_left_eval, n_right_eval] = evals;
            if expected_evaluation
                != eq_eval
                    * (d_left_eval * d_right_eval
                        + alpha * (d_left_eval * n_right_eval + n_left_eval * d_right_eval))
            {
                return Err(Error::Reject(Some(format!(
                    "depth: {}, expected evaluation not matched",
                    depth
                ))));
            }

            transcript.append_scalar(&d_left_eval);
            transcript.append_scalar(&d_right_eval);
            transcript.append_scalar(&n_left_eval);
            transcript.append_scalar(&n_right_eval);
            let r = transcript.challenge_scalar();
            point.push(r);

            Ok(FracSumClaim {
                point,
                evaluations: [
                    d_left_eval + r * (d_right_eval - d_left_eval),
                    n_left_eval + r * (n_right_eval - n_left_eval),
                ],
            })
        },
    )
}

fn run<F: JoltField>(input_num_vars: usize, table_num_vars: usize) -> Result<(), Error> {
    let (input, table) = logup_witness_gen::<F>(input_num_vars, table_num_vars);
    let mut prover_transcript = ProofTranscript::new(b"test");
    let proof = logup_prover(
        input,
        table,
        &mut prover_transcript,
        input_num_vars,
        table_num_vars,
    )?;
    let mut verifier_transcript = ProofTranscript::new(b"test");
    let _ = logup_verifier(
        proof,
        &mut verifier_transcript,
        input_num_vars,
        table_num_vars,
    )?;

    Ok(())
}

fn main() {
    let input_num_vars = 20;
    let table_num_vars = 20;
    run::<ark_bn254::Fr>(input_num_vars, table_num_vars).expect("logup failed");
}

// Start:   input_num_vars: 20, table_num_vars: 20, logup sumcheck-prove
// End:     input_num_vars: 20, table_num_vars: 20, logup sumcheck-prove ..............2.229s
// Start:   input_num_vars: 20, table_num_vars: 20, logup sumcheck-verify
// End:     input_num_vars: 20, table_num_vars: 20, logup sumcheck-verify .............6.087ms
