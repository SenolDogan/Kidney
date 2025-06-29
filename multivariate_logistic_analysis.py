import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import ast

# 1. Anlamlı değişkenleri belirle
xls = pd.ExcelFile('combo_logistic_results.xlsx')
sig_vars = set()
for k in [2,3,4,5]:
    df = pd.read_excel(xls, sheet_name=f'{k}_combo')
    for idx, row in df.iterrows():
        pvals = row['logreg_pval']
        # pvals bazen string, bazen liste olabilir
        if isinstance(pvals, str):
            try:
                pvals = ast.literal_eval(pvals)
            except Exception:
                continue
        if isinstance(pvals, (list, np.ndarray, pd.Series)):
            for v, p in zip(row['variables'].split(','), pvals):
                try:
                    if float(p) < 0.05:
                        sig_vars.add(v.strip())
                except Exception:
                    continue

sig_vars = list(sig_vars)
print('Significant variables:', sig_vars)

# 2. Modeli kur
if len(sig_vars) == 0:
    print('No significant variables found.')
else:
    df = pd.read_excel('kidney.xlsx')
    sub = df[sig_vars + ['Death']].copy()
    for col in sig_vars:
        sub[col] = sub[col].fillna(sub[col].median())
    sub = sub.dropna(subset=['Death'])
    y = sub['Death'].values
    X = sub[sig_vars].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs_const = sm.add_constant(Xs)
    logit_model = sm.Logit(y, Xs_const)
    logit_res = logit_model.fit(disp=0)
    OR = np.exp(logit_res.params[1:])
    pvals = logit_res.pvalues[1:]
    auc = roc_auc_score(y, logit_res.predict(Xs_const))
    for v, or_, p_ in zip(sig_vars, OR, pvals):
        print(f'{v}: OR={or_:.3f}, p={p_:.3g}')
    print(f'AUC: {auc:.3f}') 