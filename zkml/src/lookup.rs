use std::{collections::HashMap, hash::RandomState};

use gkr::logup::{verifier::FracSumClaim, LogupClaim, LogupInputs};
use itertools::Itertools;
use pcs::field::JoltField;

use crate::error::ZKMLError;

#[derive(Clone, Debug, Default)]
pub struct LookupWitness<F> {
    pub range: Vec<F>,
}

pub fn compute_logup_inputs<F: JoltField>(
    lookups: Vec<LookupWitness<F>>,
    table_items: Vec<F>,
) -> Result<LogupInputs<F>, ZKMLError> {
    let table_size = table_items.len();
    let table_idx = table_items
        .iter()
        .enumerate()
        .map(|x| (*x.1, x.0))
        .collect::<HashMap<F, usize, RandomState>>();
    let mut count = vec![0; table_size];
    let input_items = lookups
        .into_iter()
        .map(|w| {
            w.range
                .iter()
                .map(|x| {
                    if let Some(index) = table_idx.get(x) {
                        count[*index] += 1;
                        Ok(*x)
                    } else {
                        Err(sumcheck::error::Error::OtherError(
                            "Logup witness error".to_string(),
                        ))
                    }
                })
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;

    let count = count
        .into_iter()
        .map(|x| F::from_u64(x as u64).unwrap())
        .collect_vec();
    Ok(LogupInputs {
        table_items,
        count,
        input_items,
    })
}

#[derive(Clone, Debug, Default)]
pub struct LookupClaim<F: JoltField> {
    pub point: Vec<F>,
    pub range: F,
}

pub struct LookupTableClaim<F: JoltField> {
    pub point: Vec<F>,
    pub range: F,
    pub count: F,
}

pub fn compute_lookup_claims<F: JoltField>(
    logup_claim: LogupClaim<F>,
    lookup_challenge: F,
) -> (Vec<Option<LookupClaim<F>>>, LookupTableClaim<F>) {
    let LogupClaim {
        input_claims,
        table_claim,
    } = logup_claim;
    let input_claims = input_claims
        .into_iter()
        .map(|c| {
            c.map(|c| {
                let FracSumClaim { point, evaluations } = c;
                assert_eq!(
                    evaluations[1],
                    F::one(),
                    "The numerator should be 1, got: {:?}",
                    evaluations[1]
                );
                LookupClaim {
                    point,
                    range: evaluations[0] - lookup_challenge,
                }
            })
        })
        .collect();
    let table_claim = {
        let FracSumClaim { point, evaluations } = table_claim;

        LookupTableClaim {
            point,
            range: evaluations[0] - lookup_challenge,
            count: evaluations[1],
        }
    };
    (input_claims, table_claim)
}
