#[macro_export]
macro_rules! field_vec {
    ($f:ident) => (
        vec![]
    );
    ($f:ident, $elem:expr; $n:expr) => (
        vec![<$f as JoltField>::from_i64($elem); $n]
    );
    ($f:ident, $($x:expr),+ $(,)?) => (
        vec![ $(<$f as JoltField>::from_i64($x)),+ ]
    );
}
