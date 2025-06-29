import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('kidney.xlsx')

# Statistically significant variables selection: Use ANOVA results if available else all numerics
try:
    sig_vars = pd.read_excel('group_comparison_results.xlsx', sheet_name='anova')
    sig_vars = sig_vars[sig_vars['p_value'] < 0.05]['feature'].tolist()
    print('Significant variables from ANOVA:', sig_vars)
except Exception as e:
    print('ANOVA file not found or error:', e)
    sig_vars = [col for col in df.select_dtypes(include=[np.number]).columns if col not in ['Death']]
    print('Using all numeric columns:', sig_vars)

# Filter numeric and significant columns
sig_vars = [v for v in sig_vars if v in df.columns and df[v].dtype in [np.float64, np.int64]]
print('Final variable list:', sig_vars)

# Split by Death
df_dead = df[df['Death'] == 1][sig_vars].dropna()
df_alive = df[df['Death'] == 0][sig_vars].dropna()
print('Dead shape:', df_dead.shape, 'Alive shape:', df_alive.shape)
print('Dead columns:', df_dead.columns.tolist())
print('Alive columns:', df_alive.columns.tolist())

if df_dead.shape[0] > 0 and df_alive.shape[0] > 0:
    corr_dead = df_dead.corr()
    corr_alive = df_alive.corr()
    # Save as heatmaps
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_dead, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix - Deceased Patients')
    plt.tight_layout()
    plt.savefig('correlation_matrix_deceased.png')
    plt.close()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_alive, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix - Alive Patients')
    plt.tight_layout()
    plt.savefig('correlation_matrix_alive.png')
    plt.close()
    # Save as Excel
    with pd.ExcelWriter('correlation_matrices.xlsx') as writer:
        corr_dead.to_excel(writer, sheet_name='Deceased')
        corr_alive.to_excel(writer, sheet_name='Alive')
    print('Files saved successfully.')
else:
    print('No data for one of the groups!') 