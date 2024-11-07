# SpaZK

## Benchmarks and examples

We prepare the benchmarks for some basic blocks

1. **Sumcheck benchmarks.** (`sumcheck` is migrated from [Arkworks](https://github.com/arkworks-rs/sumcheck))
  - Run `cargo bench -p sumcheck > logs/sc.txt`.
  - To convert it into a table, run `cat logs/sc.txt | python3 scripts/bench_info.py`.
2. **PCS benchmarks.** (`pcs` is migrated from [Jolt](https://github.com/a16z/jolt))
  - Run `cargo bench -p pcs > logs/pcs.txt`.
  - To convert it into a table, run `cat pcs.txt | python3 scripts/bench_info.py`.

We also prepare the examples for linear layer implementations.

1. **Ternary matrix vector multiplication.**
  - Run `cargo run --release -p gkr --example ternary_matrix --features ark-std/print-trace > logs/ternary_matrix.txt`.
  - To convert it into a table, run `cat logs/ternary_matrix.txt | python3 sc_info.py`
2. **Normal matrix vector multiplication.**
  - Run `cargo run --release -p gkr --example normal_matrix --features ark-std/print-trace > logs/normal_matrix.txt`.
  - To convert it into a table, run `cat logs/normal_matrix.txt | python3 sc_info.py`.
3. **Ternary matrix vector multiplication implemented by GKR circuit.**
  - Run `cargo run --release -p gkr --example ternary_matrix_circuit --features ark-std/print-trace > logs/ternary_matrix_circuit.txt`.
  - To convert it into a table, run `cat logs/ternary_matrix_circuit.txt | python3 sc_info.py`.
4. **Normal matrix vector multiplication implemented by GKR circuit.**
  - Run `cargo run --release -p gkr --example normal_matrix_circuit --features ark-std/print-trace > logs/normal_matrix_circuit.txt`.
  - To convert it into a table, run `cat logs/normal_matrix_circuit.txt | python3 sc_info.py`

## ZKML
We implement a simple example containing the linear layer, relu layer and ternary sparse linear layer. Run the following command to run the example:
```
cargo run -r -p zkml --example zkml --features ark-std/print-trace
```

## Acknowledgement
The sumcheck package is modified from [arkworks-rs/sumcheck](https://github.com/arkworks-rs/sumcheck) and PCS is modified from [a16z/jolt](https://github.com/a16z/jolt/tree/main/src).