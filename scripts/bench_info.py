import sys
import re
import pandas as pd

def main():
    transcript = sys.stdin.read()
    sc_match = re.compile(r'(\S+)/(\d+)')
    pcs_match = re.compile(r'(\S+)/(\S+)/nv:(\d+)/sp:(\d+)')
    time_match = re.compile(r'time:\s+\[\d+.\d+ \S+ (\d+.\d+ \S+) \d+.\d+ \S+\]')

    sc = sc_match.findall(transcript)
    pcs = pcs_match.findall(transcript)
    times = time_match.findall(transcript)

    data = []
    if len(pcs) == 0:
        for e, t in zip(sc, times):
            method, nv = e
            data.append([method, nv, t])

        df_pivot = pd.DataFrame(data, columns = ['method', 'num_vars', 'time'])
        
    elif len(pcs) > 0:
        for e, t in zip(pcs, times):
            method, pcs, nv, sp = e
            sp = "1/" + str(1 << int(sp))
            data.append([pcs, nv, sp, method, t])

        df = pd.DataFrame(data, columns= ['pcs', 'num_vars', 'sparsity', 'method', 'time'])
        pcs = df['pcs'].unique()
        num_vars = df['num_vars'].unique()
        sparsity = df['sparsity'].unique()
        method = df['method'].unique()

        df['method'] = pd.Categorical(df['method'], categories=method, ordered=True)

        df_idx = pd.MultiIndex.from_product([pcs, num_vars, sparsity], names=['pcs', 'num_vars', 'sparsity'])
        df_pivot = df.pivot(index = ('pcs', 'num_vars', 'sparsity'), columns = 'method', values = 'time')
        df_pivot = df_pivot.reindex(df_idx).reset_index()

    df_pivot.to_csv(sys.stdout, index=False, sep=',')

if __name__ == '__main__':
    main()