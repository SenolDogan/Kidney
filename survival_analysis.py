import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import os

# Veriyi oku
df = pd.read_excel('kidney.xlsx')

# Survival süresi ve event sütunlarını sayısal yap
# Hatalı değerleri NaN yap, sonra satırları at

duration_col = 'Time_to_death_after_baseline_months'
event_col = 'Death'
df[duration_col] = pd.to_numeric(df[duration_col], errors='coerce')
df[event_col] = pd.to_numeric(df[event_col], errors='coerce')
df = df.dropna(subset=[duration_col, event_col])
df[event_col] = df[event_col].astype(int)

# Diğer değişkenler için KM plotlar (önceki kod)
exclude = [duration_col, event_col]
variables = [col for col in df.columns if col not in exclude]

# Plotları kaydedeceğimiz klasörü oluştur
os.makedirs('plots', exist_ok=True)

for var in variables:
    plt.figure()
    # Kategorik mi sürekli mi kontrol et
    if df[var].dtype == 'object' or df[var].nunique() <= 5:
        groups = df[var].unique()
        if len(groups) != 2:
            plt.close()
            continue  # Sadece iki grup için p-value ve HR hesaplanabilir
        mask1 = df[var] == groups[0]
        mask2 = df[var] == groups[1]
        label1 = f"{var}={groups[0]}"
        label2 = f"{var}={groups[1]}"
    else:
        median = df[var].median()
        mask1 = df[var] <= median
        mask2 = df[var] > median
        label1 = f"{var}≤{median:.2f}"
        label2 = f"{var}>{median:.2f}"

    # KM plot
    kmf = KaplanMeierFitter()
    kmf.fit(df[mask1][duration_col], event_observed=df[mask1][event_col], label=label1)
    kmf.plot_survival_function()
    kmf.fit(df[mask2][duration_col], event_observed=df[mask2][event_col], label=label2)
    kmf.plot_survival_function()

    # Log-rank test
    try:
        results = logrank_test(
            df[mask1][duration_col], df[mask2][duration_col],
            event_observed_A=df[mask1][event_col], event_observed_B=df[mask2][event_col]
        )
        p_value = results.p_value
    except Exception:
        p_value = None

    # CoxPH ile HR
    try:
        temp = df[[duration_col, event_col, var]].copy()
        if temp[var].dtype != 'object' and temp[var].nunique() > 2:
            temp[var] = (temp[var] > median).astype(int)
        elif temp[var].dtype == 'object':
            temp[var] = (temp[var] == groups[1]).astype(int)
        cph = CoxPHFitter()
        cph.fit(temp, duration_col=duration_col, event_col=event_col)
        hr = cph.hazard_ratios_[var]
    except Exception:
        hr = None

    plt.title(f"KM by {var}\n"
              f"p-value: {p_value:.4f}  HR: {hr:.2f}" if p_value is not None and hr is not None else
              f"KM by {var}\n(p-value veya HR hesaplanamadı)")
    plt.xlabel("Time (months)")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/KM_{var}.png")
    plt.close()

# Death değişkenine göre iki grup için KM plotu (event ile grup aynı olduğu için anlamsızdır, eklenmedi) 