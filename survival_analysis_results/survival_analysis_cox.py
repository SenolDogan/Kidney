import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter

# Veri yükle
file_path = 'kidney.xlsx'
df = pd.read_excel(file_path)

# Gerekli kolonlar: Time_to_death_after_baseline_months, Death
# Eksik olanları at
# (ilk başta at, sonra kalan NaN'leri doldur)
df = df[df['Time_to_death_after_baseline_months'].notnull() & df['Death'].notnull()]

# Temel klinik değişkenler
features = [
    'AGE_Baseline',
    'Sex_1male_2female',
    'Diabetes_status_1yes_0no',
    'eGFR_CKD_EPI_Creatinine_at_Baseline',
    'BMI'
]

# Eksik değerleri median ile doldur
for col in features:
    df[col] = df[col].fillna(df[col].median())

# Kategorik değişkenleri uygun şekilde dönüştür
cat_cols = ['Sex_1male_2female', 'Diabetes_status_1yes_0no']
df[cat_cols] = df[cat_cols].astype('category')
df_encoded = pd.get_dummies(df[features], columns=cat_cols, drop_first=True)

# Survival kolonlarını da ekle
data = pd.concat([df_encoded, df[['Time_to_death_after_baseline_months','Death']].reset_index(drop=True)], axis=1)

# Tüm dataframe'de kalan NaN'leri median ile doldur
for col in data.columns:
    if data[col].isnull().any():
        data[col] = data[col].fillna(data[col].median())

# Cox regresyonu
cph = CoxPHFitter()
cph.fit(data, duration_col='Time_to_death_after_baseline_months', event_col='Death')
summary = cph.summary
summary.to_excel('survival_analysis_results/cox_regression_summary.xlsx')

# En önemli risk faktörleri (p<0.05)
sig = summary[summary['p'] < 0.05].sort_values('p')
sig.to_excel('survival_analysis_results/cox_significant_factors.xlsx')

# Forest plot (HR ve CI)
plt.figure(figsize=(8, min(10, len(sig)*0.5+2)))
plt.errorbar(sig['exp(coef)'], range(len(sig)), xerr=[sig['exp(coef)']-sig['exp(coef) lower 95%'], sig['exp(coef) upper 95%']-sig['exp(coef)']], fmt='o')
plt.yticks(range(len(sig)), sig.index)
plt.axvline(1, color='red', linestyle='--')
plt.xlabel('Hazard Ratio (HR)')
plt.title('Cox Regression: Significant Risk Factors')
plt.tight_layout()
plt.savefig('survival_analysis_results/cox_forestplot.png')
plt.close()

# Kaplan-Meier plot for top 2 significant factors (median split)
kmf = KaplanMeierFitter()
for var in sig.index[:2]:
    # Dummy ise orijinal değişkeni bul
    base_var = var.split('_')[0]
    if var == 'Diabetes_status_1yes_0no_1':
        mask_true = data[var] > 0.5
        mask_false = ~mask_true
        labels = ['Diabetes=1', 'Diabetes=0']
        masks = [mask_true, mask_false]
        # HR ve p-value al
        hr = summary.loc[var, 'exp(coef)']
        pval = summary.loc[var, 'p']
    elif base_var in df.columns and df[base_var].nunique() > 2:
        median = df[base_var].median()
        mask_high = df[base_var] > median
        mask_low = ~mask_high
        labels = [f'{base_var} > {median}', f'{base_var} <= {median}']
        masks = [mask_high, mask_low]
        hr = summary.loc[var, 'exp(coef)']
        pval = summary.loc[var, 'p']
    else:
        mask_true = data[var] > 0.5
        mask_false = ~mask_true
        labels = [f'{var}=1', f'{var}=0']
        masks = [mask_true, mask_false]
        hr = summary.loc[var, 'exp(coef)']
        pval = summary.loc[var, 'p']
    plt.figure(figsize=(7,5))
    for label, mask in zip(labels, masks):
        kmf.fit(df.loc[mask, 'Time_to_death_after_baseline_months'], df.loc[mask, 'Death'], label=label)
        kmf.plot_survival_function(ci_show=False)
    plt.title(f'Kaplan-Meier: {var}\nHR={hr:.2f}, p={pval:.3g}')
    plt.xlabel('Months')
    plt.ylabel('Survival Probability')
    plt.tight_layout()
    plt.savefig(f'survival_analysis_results/km_{var}.png')
    plt.close()

print('Cox regresyonu (temel klinik değişkenlerle) tamamlandı. Sonuçlar ve grafikler survival_analysis_results klasörüne kaydedildi.') 