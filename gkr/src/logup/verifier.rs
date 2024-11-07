use pcs::{field::JoltField, poly::eq_poly::EqPolynomial, utils::transcript::ProofTranscript};
use sumcheck::{
    error::Error,
    protocol::{verifier::SubClaim, PolynomialInfo},
    MLSumcheck,
};

use super::prover::FracSumProof;

#[derive(Clone, Debug, PartialEq)]
pub struct FracSumClaim<F: JoltField> {
    pub point: Vec<F>,
    /// [Denominator evaluation, numerator evaluation]
    pub evaluations: [F; 2],
}

pub fn frac_sum_verifier<F: JoltField>(
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

            let eq_eval = EqPolynomial::new(last_point).evaluate(&point);
            let [d_left_eval, d_right_eval, n_left_eval, n_right_eval] = evals;
            let got = eq_eval
                * (d_left_eval * d_right_eval
                    + alpha * (d_left_eval * n_right_eval + n_left_eval * d_right_eval));
            if expected_evaluation != got {
                return Err(Error::Reject(Some(format!(
                    "depth: {}, expected evaluation not matched, want: {:?}, got: {:?}",
                    depth, expected_evaluation, got,
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
