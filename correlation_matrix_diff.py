import pandas as pd
import numpy as np
from scipy.stats import t
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel('kidney.xlsx')

# Get variable list from previous correlation matrix
variables = [
    'eGFR_CKD_EPI_Creatinine_at_Baseline', 'A_Body_Shape_Index_ABSI', 'WWI_Weight_adjusted_Waist_Index',
    'AGE_Baseline', 'ConI_Conicity_Index', 'WHR', 'eTBF_estimated_Total_Body_Fat', 'BRI_Body_Roundness_Index',
    'WHtR', 'WAIST_circumference_cm', 'AVI_Abdominal_Volume_Index', 'Urinary_Albumin_Creatinine_ratio_mg_g',
    'Lipid_accumulation_product_LAP', 'Visceral_adiposity_index_VAI'
]

# Split by group
df_dead = df[df['Death'] == 1]
df_alive = df[df['Death'] == 0]

def correlation_pvalue_matrix(data, variables):
    corr = data[variables].corr()
    pval = pd.DataFrame(np.ones(corr.shape), columns=variables, index=variables)
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i >= j:
                continue
            x = data[var1]
            y = data[var2]
            valid = x.notna() & y.notna()
            n = valid.sum()
            if n > 2:
                r = corr.loc[var1, var2]
                t_stat = r * np.sqrt((n-2)/(1-r**2)) if abs(r) < 1 else np.nan
                p = 2 * (1 - t.cdf(abs(t_stat), df=n-2)) if not np.isnan(t_stat) else np.nan
                pval.loc[var1, var2] = p
                pval.loc[var2, var1] = p
            else:
                pval.loc[var1, var2] = np.nan
                pval.loc[var2, var1] = np.nan
    return pval

# Calculate p-value matrices
pval_dead = correlation_pvalue_matrix(df_dead, variables)
pval_alive = correlation_pvalue_matrix(df_alive, variables)

# Overwrite correlation_matrices.xlsx with p-value matrices
with pd.ExcelWriter('correlation_matrices.xlsx') as writer:
    pval_dead.to_excel(writer, sheet_name='Deceased')
    pval_alive.to_excel(writer, sheet_name='Alive')

def plot_pvalue_heatmap(pval_matrix, title, filename):
    plt.figure(figsize=(12,10))
    ax = sns.heatmap(pval_matrix, annot=False, fmt='.2g', cmap='viridis', cbar_kws={'label': 'p-value'})
    for i in range(len(pval_matrix)):
        for j in range(len(pval_matrix)):
            p = pval_matrix.iloc[i, j]
            if np.isnan(p):
                text = ''
            else:
                text = f'{p:.2g}'
            color = 'red' if not np.isnan(p) and p < 0.05 else 'black'
            ax.text(j+0.5, i+0.5, text, ha='center', va='center', color=color, fontsize=10)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot with significant p-values in red
plot_pvalue_heatmap(pval_dead, 'Correlation Matrix Deceased (p-values)', 'correlation_matrix_deceased_pvalue.png')
plot_pvalue_heatmap(pval_alive, 'Correlation Matrix Alive (p-values)', 'correlation_matrix_alive_pvalue.png') 