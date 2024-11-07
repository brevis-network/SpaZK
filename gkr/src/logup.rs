use itertools::Itertools;
use pcs::{
    field::JoltField,
    utils::{math::Math, transcript::ProofTranscript},
};
use prover::{frac_sum_prover, FracSumProof};
use sumcheck::error::Error;
use verifier::{frac_sum_verifier, FracSumClaim};

pub mod prover;
pub mod verifier;

pub struct LogupInputs<F> {
    pub table_items: Vec<F>,
    pub count: Vec<F>,
    pub input_items: Vec<Vec<F>>,
}

pub struct LogupWitness<F> {
    pub input_trees: Vec<Vec<Vec<Vec<F>>>>,
    pub table_tree: Vec<Vec<Vec<F>>>,
}

pub fn logup_witness_gen_multi_input<F: JoltField>(
    logup_inputs: &LogupInputs<F>,
    challenge: F,
) -> Result<LogupWitness<F>, Error> {
    let LogupInputs {
        table_items,
        count,
        input_items,
    } = logup_inputs;

    assert!(!input_items.is_empty());

    let table = table_items.iter().map(|x| *x + challenge).collect_vec();
    let table_size = table.len();

    let inputs = input_items
        .iter()
        .map(|input_items| input_items.iter().map(|x| *x + challenge).collect_vec())
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

    let table_num_vars = table_size.log_2();
    let table_tree = {
        let mut table_tree = vec![vec![vec![], vec![]]; table_num_vars + 1];
        table_tree[table_num_vars] = vec![table, count.clone()];
        compute_frac_tree(&mut table_tree, table_num_vars);
        table_tree
    };

    let input_trees = {
        inputs
            .into_iter()
            .map(|input| {
                if input.is_empty() {
                    return vec![];
                }
                let input_size = input.len();
                let input_num_vars = input_size.log_2();

                let mut input_tree = vec![vec![vec![], vec![]]; input_num_vars + 1];
                input_tree[input_num_vars] = vec![input, vec![F::one(); input_size]];
                compute_frac_tree(&mut input_tree, input_num_vars);
                input_tree
            })
            .collect_vec()
    };

    Ok(LogupWitness {
        input_trees,
        table_tree,
    })
}

pub struct LogupProof<F: JoltField> {
    pub input_proofs: Vec<Option<FracSumProof<F>>>,
    pub table_proof: FracSumProof<F>,
}

pub struct LogupProverOutput<F: JoltField> {
    pub proof: LogupProof<F>,
    pub input_points: Vec<Vec<F>>,
    pub table_point: Vec<F>,
}

pub fn logup_prover<F: JoltField>(
    logup_witness: &LogupWitness<F>,
    transcript: &mut ProofTranscript,
) -> Result<LogupProverOutput<F>, Error> {
    let LogupWitness {
        input_trees,
        table_tree,
    } = logup_witness;
    let mut input_proofs = Vec::with_capacity(input_trees.len());
    let mut input_points = Vec::with_capacity(input_trees.len());
    for input_tree in input_trees.iter() {
        if input_tree.is_empty() {
            input_proofs.push(None);
            input_points.push(vec![]);
        } else {
            let input_sum = [input_tree[0][0][0], input_tree[0][1][0]];
            let (proof, point) = frac_sum_prover(input_sum, input_tree, transcript)?;
            input_proofs.push(Some(proof));
            input_points.push(point);
        }
    }
    let table_sum = [table_tree[0][0][0], table_tree[0][1][0]];
    let (table_proof, table_point) = frac_sum_prover(table_sum, table_tree, transcript)?;
    Ok(LogupProverOutput {
        proof: LogupProof {
            input_proofs,
            table_proof,
        },
        input_points,
        table_point,
    })
}

pub struct LogupClaim<F: JoltField> {
    pub input_claims: Vec<Option<FracSumClaim<F>>>,
    pub table_claim: FracSumClaim<F>,
}

pub fn logup_verifier<F: JoltField>(
    proof: LogupProof<F>,
    transcript: &mut ProofTranscript,
) -> Result<LogupClaim<F>, Error> {
    let input_claims = proof
        .input_proofs
        .into_iter()
        .map(|input_proof| {
            input_proof.map(|input_proof| frac_sum_verifier(input_proof, transcript).unwrap())
        })
        .collect::<Vec<_>>();
    let table_claim = frac_sum_verifier(proof.table_proof, transcript)?;
    Ok(LogupClaim {
        input_claims,
        table_claim,
    })
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use itertools::Itertools;
    use pcs::{
        field::JoltField, field_vec, poly::dense::DenseMultilinearExtension,
        utils::transcript::ProofTranscript,
    };
    use rand::{rngs::OsRng, Rng};

    use crate::logup::{
        logup_prover, logup_witness_gen_multi_input, verifier::FracSumClaim, LogupInputs,
    };

    use super::{logup_verifier, LogupClaim, LogupProverOutput};

    fn run_logup<F: JoltField>(
        table_items: Vec<F>,
        input_items: Vec<Vec<F>>,
        count: Vec<F>,
        challenge: F,
    ) {
        let logup_inputs = LogupInputs {
            table_items,
            input_items,
            count,
        };
        let wit = logup_witness_gen_multi_input(&logup_inputs, challenge)
            .expect("logup witness generation failed");

        let input_sums = wit
            .input_trees
            .iter()
            .map(|x| (x[0][0][0], x[0][1][0]))
            .collect_vec();
        let table_sum = (wit.table_tree[0][0][0], wit.table_tree[0][1][0]);
        let sum = input_sums
            .iter()
            .fold((table_sum.0, -table_sum.1), |(acc_d, acc_n), (d, n)| {
                (acc_d * d, acc_d * n + acc_n * d)
            });
        assert_eq!(sum.1, F::zero());

        let mut prover_transcript = ProofTranscript::new(b"logup");
        let LogupProverOutput {
            proof,
            input_points,
            table_point,
        } = logup_prover(&wit, &mut prover_transcript).expect("logup prover failed");

        let mut verifier_transcript = ProofTranscript::new(b"logup");
        let LogupClaim {
            input_claims,
            table_claim,
        } = logup_verifier(proof, &mut verifier_transcript).expect("logup verifier failed");

        let LogupInputs {
            input_items,
            table_items,
            ..
        } = logup_inputs;

        input_items
            .into_iter()
            .zip(input_points)
            .zip(input_claims)
            .for_each(|((input_items, input_point), input_claim)| {
                let num_vars = input_point.len();
                let expected_input_claim = if input_items.is_empty() {
                    None
                } else {
                    let poly =
                        DenseMultilinearExtension::from_evaluations_vec(num_vars, input_items);
                    let eval = poly.evaluate(&input_point) + challenge;
                    Some(FracSumClaim {
                        point: input_point,
                        evaluations: [eval, F::one()],
                    })
                };
                assert_eq!(input_claim, expected_input_claim);
            });

        let table_poly =
            DenseMultilinearExtension::from_evaluations_vec(table_point.len(), table_items);
        let expected_table_eval = table_poly.evaluate(&table_point) + challenge;
        assert_eq!(table_claim.evaluations[0], expected_table_eval);
    }

    #[test]
    fn test_logup_simple() {
        let table = field_vec![Fr, 0, 1, 2, 3];
        let input = vec![field_vec![Fr, 1], field_vec![Fr, 2]];
        let challenge = Fr::from_i64(97);
        run_logup(table, input, field_vec![Fr, 0, 1, 1, 0], challenge);
    }

    #[test]
    fn test_logup() {
        let num_vars = 4;

        let table_num_vars = num_vars - 1;
        let input_num_vars = num_vars - 1;

        let mut csprng = OsRng;
        let table_size = 1 << table_num_vars;
        let table = (0..table_size)
            .map(|_| Fr::random(&mut csprng))
            .collect_vec();

        let input_size = 1 << input_num_vars;
        let mut count = vec![0; table_size];
        let input = (0..4)
            .map(|_| {
                (0..input_size)
                    .map(|_| {
                        let index = csprng.gen_range(0..table_size);
                        count[index] += 1;
                        table[index]
                    })
                    .collect_vec()
            })
            .collect_vec();

        let challenge = Fr::from_i64(97);
        let count = count.into_iter().map(|x| x.into()).collect_vec();
        run_logup(table, input, count, challenge);
    }
}
