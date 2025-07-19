import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_excel('kidney.xlsx')
if 'Death' not in df.columns:
    raise ValueError('Death column not found!')

# Select numeric variables (excluding Death)
num_vars = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'Death']
results = []
for var in num_vars:
    sub = df[[var, 'Death']].copy()
    sub[var] = sub[var].fillna(sub[var].median())
    sub = sub.dropna(subset=['Death'])
    y = sub['Death'].values
    X = sub[[var]].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs_const = sm.add_constant(Xs)
    try:
        logit_model = sm.Logit(y, Xs_const)
        logit_res = logit_model.fit(disp=0)
        OR = np.exp(logit_res.params[1])
        pval = logit_res.pvalues[1]
        auc = roc_auc_score(y, logit_res.predict(Xs_const))
        results.append({'variable': var, 'OR': OR, 'p_value': pval, 'AUC': auc})
    except Exception as e:
        continue

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('p_value')
print(results_df)
results_df.to_excel('univariate_logistic_results.xlsx', index=False) 