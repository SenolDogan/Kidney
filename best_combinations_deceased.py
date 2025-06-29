import pandas as pd
import numpy as np
from itertools import combinations

# Load p-value matrix for deceased patients
pval_df = pd.read_excel('correlation_matrices.xlsx', sheet_name='Deceased', index_col=0)
variables = pval_df.columns.tolist()

results = {}
for k in [2, 3, 4, 5]:
    best_combos = []
    for combo in combinations(variables, k):
        sub = pval_df.loc[list(combo), list(combo)]
        mask = np.triu(np.ones((k, k)), 1).astype(bool)
        pvals = sub.values[mask]
        if len(pvals) > 0:
            mean_p = np.nanmean(pvals)
            # List all pairwise p-values as (var1, var2, p) tuples
            pairs = list(combinations(combo, 2))
            pair_pvals = []
            for idx, (var1, var2) in enumerate(pairs):
                p = pval_df.loc[var1, var2]
                pair_pvals.append(f"({var1}, {var2}): {p:.3g}")
            best_combos.append((combo, mean_p, '; '.join(pair_pvals)))
    best_combos = sorted(best_combos, key=lambda x: x[1])[:5]
    results[k] = best_combos

# Save to Excel
with pd.ExcelWriter('best_combinations_deceased.xlsx') as writer:
    for k in results:
        df = pd.DataFrame([
            {'variables': ', '.join(combo), 'mean_p_value': mean_p, 'pair_p_values': pair_pvals}
            for combo, mean_p, pair_pvals in results[k]
        ])
        df.to_excel(writer, sheet_name=f'{k}_combo', index=False) 