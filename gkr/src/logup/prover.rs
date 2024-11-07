use std::rc::Rc;

use pcs::{
    field::JoltField,
    poly::dense::DenseMultilinearExtension,
    utils::{precompute_eq, transcript::ProofTranscript},
};
use sumcheck::{error::Error, protocol::ListOfProductsOfPolynomials, MLSumcheck};

pub struct FracSumProof<F: JoltField> {
    pub sum: [F; 2],
    pub sumcheck_proofs: Vec<(sumcheck::Proof<F>, [F; 4])>,
}

pub fn frac_sum_prover<F: JoltField>(
    sum: [F; 2],
    tree_cells: &[Vec<Vec<F>>],
    transcript: &mut ProofTranscript,
) -> Result<(FracSumProof<F>, Vec<F>), Error> {
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
    Ok((
        FracSumProof {
            sum,
            sumcheck_proofs,
        },
        point,
    ))
}
