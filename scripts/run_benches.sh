echo "bench normal_matrix"
cargo run --release  --package gkr --example normal_matrix --features ark-std/print-trace > logs/normal_matrix.txt
echo "bench ternary matrix"
cargo run --release  --package gkr --example ternary_matrix --features ark-std/print-trace > logs/ternary_matrix.txt
echo "bench normal_matrix_circuit"
cargo run --release  --package gkr --example normal_matrix_circuit --features ark-std/print-trace > logs/normal_matrix_circuit.txt
echo "bench ternary_matrix_circuit"
cargo run --release  --package gkr --example ternary_matrix_circuit --features ark-std/print-trace > logs/ternary_matrix_circuit.txt
echo "bench normal_matrix_libra_circuit"
cargo run --release --package gkr --example normal_matrix_libra_circuit --features ark-std/print-trace > logs/normal_matrix_libra_circuit.txt