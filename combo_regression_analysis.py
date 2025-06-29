import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Load data
df = pd.read_excel('kidney.xlsx')
# Use only rows with non-missing Death
if 'Death' not in df.columns:
    raise ValueError('Death column not found!')

# Load best combinations
def get_combos(sheet):
    combos = []
    dfc = pd.read_excel('best_combinations_deceased.xlsx', sheet_name=sheet)
    for v in dfc['variables']:
        combos.append([x.strip() for x in v.split(',')])
    return combos

results = {}
for k in [2, 3, 4, 5]:
    combos = get_combos(f'{k}_combo')
    rows = []
    for combo in combos:
        # Fill missing values with median for each variable in combo
        sub = df[combo + ['Death']].copy()
        for col in combo:
            sub[col] = sub[col].fillna(sub[col].median())
        sub = sub.dropna(subset=['Death'])
        y = sub['Death'].values
        if len(np.unique(y)) < 2:
            rows.append({
                'variables': ', '.join(combo),
                'logreg_OR': 'SKIPPED (only one class)',
                'logreg_pval': 'SKIPPED',
                'logreg_auc': 'SKIPPED'
            })
            continue
        X = sub[combo].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        # Logistic Regression (statsmodels for p-value)
        Xs_const = sm.add_constant(Xs)
        logit_model = sm.Logit(y, Xs_const)
        try:
            logit_res = logit_model.fit(disp=0)
            OR = np.exp(logit_res.params[1:])
            logreg_pvals = logit_res.pvalues[1:]
            auc = roc_auc_score(y, logit_res.predict(Xs_const))
        except Exception as e:
            OR = [np.nan]*len(combo)
            logreg_pvals = [np.nan]*len(combo)
            auc = np.nan
        rows.append({
            'variables': ', '.join(combo),
            'logreg_OR': OR,
            'logreg_pval': logreg_pvals,
            'logreg_auc': auc
        })
    results[k] = pd.DataFrame(rows)

with pd.ExcelWriter('combo_logistic_results.xlsx') as writer:
    for k in results:
        results[k].to_excel(writer, sheet_name=f'{k}_combo', index=False) 