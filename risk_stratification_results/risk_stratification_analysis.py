import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from scipy.stats import ttest_ind

def run_univariate(df):
    # Outcome grupları oluştur
    def outcome_group(row):
        if row['Death'] == 1:
            if row['Time_to_death_after_baseline_months'] <= 12:
                return 'Died <1yr'
            else:
                return 'Died >1yr'
        else:
            return 'Alive'
    df['Outcome_Group'] = df.apply(outcome_group, axis=1)
    # Yaş grupları oluştur
    bins = [0, 40, 60, np.inf]
    labels = ['0-40', '41-60', '61+']
    df['Age_Group'] = pd.cut(df['AGE_Baseline'], bins=bins, labels=labels, right=False)
    # Univariate analiz
    exclude = ['NUMMER','SERUMNO','Baseline_DATE_study_entry','Datum_Erstdialyse_DD.MM.YYYY','Birth_DATE','Date_Follow_up_MM.YYYY','Date_Follow_Up','Date_of_Death_DD.MM.YYYY','Death','Time_to_death_after_baseline_months','Outcome_Group','Age_Group']
    features = [col for col in df.columns if col not in exclude and df[col].dtype != 'O']
    results = []
    for col in features:
        try:
            group1 = df[df['Outcome_Group']=='Alive'][col].dropna()
            group2 = df[df['Outcome_Group']!='Alive'][col].dropna()
            stat, p = ttest_ind(group1, group2, equal_var=False)
            results.append({'Feature': col, 'p_value': p, 'Alive_mean': group1.mean(), 'Died_mean': group2.mean()})
        except Exception as e:
            results.append({'Feature': col, 'p_value': np.nan, 'Alive_mean': np.nan, 'Died_mean': np.nan})
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values('p_value')
    significant = res_df[res_df['p_value'] < 0.05]
    not_significant = res_df[res_df['p_value'] >= 0.05]
    significant.to_excel('risk_stratification_results/clinically_significant_features.xlsx', index=False)
    not_significant.to_excel('risk_stratification_results/clinically_not_significant_features.xlsx', index=False)
    print('Univariate analiz tamamlandı.')
    return df

def run_combinations(df):
    # Klinik olarak anlamlı parametreleri dosyadan oku
    sig_df = pd.read_excel('risk_stratification_results/clinically_significant_features.xlsx')
    sig_features = sig_df['Feature'].tolist()
    labels = ['0-40', '41-60', '61+']
    combo_results = []
    for age_group in labels:
        for sex in [1, 2]:
            sub = df[(df['Age_Group'] == age_group) & (df['Sex_1male_2female'] == sex)]
            if len(sub) < 10:
                continue
            for feat in sig_features:
                try:
                    median = sub[feat].median()
                    high = sub[sub[feat] > median]
                    low = sub[sub[feat] <= median]
                    high_dead = (high['Outcome_Group'] != 'Alive').mean()
                    low_dead = (low['Outcome_Group'] != 'Alive').mean()
                    combo_results.append({
                        'Age_Group': age_group,
                        'Sex': 'Male' if sex==1 else 'Female',
                        'Feature': feat,
                        'High_group_dead_rate': high_dead,
                        'Low_group_dead_rate': low_dead,
                        'N': len(sub)
                    })
                except Exception as e:
                    continue
    combo_df = pd.DataFrame(combo_results)
    combo_df.to_excel('risk_stratification_results/feature_combinations_by_age_sex.xlsx', index=False)
    pivot = combo_df.pivot_table(index=['Age_Group','Sex','Feature'], values=['High_group_dead_rate','Low_group_dead_rate','N'])
    pivot.to_excel('risk_stratification_results/feature_combinations_summary.xlsx')
    gk = combo_df[(combo_df['Age_Group']=='0-40') & (combo_df['Sex']=='Female')]
    gk = gk.sort_values('High_group_dead_rate', ascending=False)
    plt.figure(figsize=(8,5))
    plt.barh(gk['Feature'], gk['High_group_dead_rate'])
    plt.xlabel('Death Rate (High group)')
    plt.title('0-40 Age, Female: Death Rate by Feature (High group)')
    plt.tight_layout()
    plt.savefig('risk_stratification_results/death_rate_0_40_female.png')
    plt.close()
    print('Kombinasyon analizleri ve görseller tamamlandı.')
    # Her yaş ve cinsiyet grubunda en etkili 5 parametreyi tablo ve bar plot olarak kaydet
    for age_group in ['0-40', '41-60', '61+']:
        for sex in ['Female', 'Male']:
            sub = combo_df[(combo_df['Age_Group']==age_group) & (combo_df['Sex']==sex)]
            if sub.empty:
                continue
            sub = sub.sort_values('High_group_dead_rate', ascending=False).head(5)
            # Tablo olarak kaydet
            sub.to_excel(f'risk_stratification_results/top5_deathrate_{age_group}_{sex}.xlsx', index=False)
            # Bar plot
            plt.figure(figsize=(8,5))
            plt.barh(sub['Feature'], sub['High_group_dead_rate'])
            plt.xlabel('Death Rate (High group)')
            plt.title(f'{age_group}, {sex}: Top 5 Features by Death Rate (High group)')
            plt.tight_layout()
            plt.savefig(f'risk_stratification_results/top5_deathrate_{age_group}_{sex}.png')
            plt.close()

if __name__ == '__main__':
    df = pd.read_excel('kidney.xlsx')
    if not os.path.exists('risk_stratification_results/clinically_significant_features.xlsx'):
        df = run_univariate(df)
    else:
        # Outcome ve Age_Group kolonları yoksa ekle
        if 'Outcome_Group' not in df.columns or 'Age_Group' not in df.columns:
            df = run_univariate(df)
    run_combinations(df) 