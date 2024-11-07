import re
import pandas as pd
import sys

def main():
    transcript = sys.stdin.read()
    end_pattern = re.compile(r"End:\s+(\d+), (\d+/\d+): (\S+ \S+) \S+ (\S+)\s+\.+(\S+)")
    
    data = []
    lines = transcript.split("\n")
    for line in lines:
        end_match = end_pattern.match(line)
        if end_match:
            log_n, sparsity, exp_type, step_type, time = end_match.groups()
            data.append((log_n, sparsity, step_type, time))

    # Create DataFrame
    df = pd.DataFrame(data, columns=["log_n", "sparsity", "type", "time"])
    log_n_order = df["log_n"].unique()
    sparsity_order = df["sparsity"].unique()
    type_order = df["type"].unique()

    df['type'] = pd.Categorical(df['type'], categories=type_order, ordered=True)

    # Pivot the DataFrame
    df_pivot = df.pivot(index=["log_n", "sparsity"], columns="type", values="time")

    idx = pd.MultiIndex.from_product([log_n_order, sparsity_order], names=['log_n', 'sparsity'])
    df_pivot = df_pivot.reindex(idx).reset_index()

    print(df_pivot)

if __name__ == "__main__":
    main()
