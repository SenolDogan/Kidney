import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# Veriyi yükle
file_path = 'kidney.xlsx'
df = pd.read_excel(file_path)

# Sadece iki grubu al
mask = df['Death'].notnull() & df['Time_to_death_after_baseline_months'].notnull()
df = df[mask].copy()
df['SurvivalGroup'] = np.select(
    [
        (df['Death'] == 1) & (df['Time_to_death_after_baseline_months'] <= 12),
        (df['Death'] == 1) & (df['Time_to_death_after_baseline_months'] > 12)
    ],
    ['Died_within_1yr', 'Died_after_1yr'],
    default='Other'
)
df = df[df['SurvivalGroup'].isin(['Died_within_1yr', 'Died_after_1yr'])].copy()

# Analiz yapılacak değişkenler
exclude = [
    'NUMMER','SERUMNO','Baseline_DATE_study_entry','Datum_Erstdialyse_DD.MM.YYYY','Birth_DATE',
    'Date_Follow_up_MM.YYYY','Date_Follow_Up','Date_of_Death_DD.MM.YYYY','Death',
    'Time_to_death_after_baseline_months','SurvivalGroup'
]
variables = [col for col in df.columns if col not in exclude and df[col].dtype != 'O']

results = []
for var in variables:
    # Median split
    median = df[var].median()
    df['Group'] = np.where(df[var] > median, f'{var}>med', f'{var}<=med')
    # Kaplan-Meier
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(7,5))
    for g in df['Group'].unique():
        ix = df['Group'] == g
        kmf.fit(df.loc[ix, 'Time_to_death_after_baseline_months'], df.loc[ix, 'Death'], label=g)
        kmf.plot_survival_function(ci_show=False)
    plt.title(f'Kaplan-Meier: {var} (median split)')
    plt.xlabel('Months')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'km_{var}_2groups_median.png')
    plt.close()
    # Log-rank testi
    ix1 = df['Group'] == f'{var}>med'
    ix2 = df['Group'] == f'{var}<=med'
    if ix1.sum() > 0 and ix2.sum() > 0:
        lr = logrank_test(
            df.loc[ix1, 'Time_to_death_after_baseline_months'],
            df.loc[ix2, 'Time_to_death_after_baseline_months'],
            df.loc[ix1, 'Death'],
            df.loc[ix2, 'Death']
        )
        logrank_p = lr.p_value
    else:
        logrank_p = np.nan
    # Cox regresyonu
    cph_df = df[[var, 'Time_to_death_after_baseline_months', 'Death']].dropna()
    cph_df['Group'] = np.where(cph_df[var] > median, 1, 0)
    cph = CoxPHFitter()
    try:
        cph.fit(cph_df[['Group', 'Time_to_death_after_baseline_months', 'Death']], duration_col='Time_to_death_after_baseline_months', event_col='Death')
        hr = cph.hazard_ratios_['Group']
        pval = cph.summary.loc['Group', 'p']
    except Exception as e:
        hr, pval = np.nan, np.nan
    results.append({'variable': var, 'median': median, 'cox_HR': hr, 'cox_p': pval, 'logrank_p': logrank_p})

# Sonuçları kaydet
results_df = pd.DataFrame(results)
results_df.to_excel('2group_survival_results.xlsx', index=False)
print('Tüm değişkenler için 2 grup survival analizleri tamamlandı. Sonuçlar 2group_survival_results.xlsx dosyasına kaydedildi.') 