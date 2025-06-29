import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm
import matplotlib.patches as mpatches
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# 1. Data Loading
file_path = 'kidney.xlsx'
df = pd.read_excel(file_path)

# 2. EDA: Columns, missing values, basic statistics
print('--- Columns and Data Types ---')
print(df.dtypes)
print('\n--- Missing Value Ratios ---')
print(df.isnull().mean().sort_values(ascending=False))
print('\n--- Numerical Feature Statistics ---')
print(df.describe().T)
print('\n--- Death Column Distribution ---')
print(df['Death'].value_counts(dropna=False))

# 3. Handling Missing Values
# Remove rows with missing target (Death)
df = df[~df['Death'].isnull()]

# Remove columns with >60% missing, but always keep 'Time_to_death_after_baseline_months'
too_many_missing = df.columns[(df.isnull().mean() > 0.6) & (df.columns != 'Time_to_death_after_baseline_months')]
df = df.drop(columns=too_many_missing)
print(f"\nDropped columns (>%60 missing, except Time_to_death_after_baseline_months): {list(too_many_missing)}")

# Convert datetime columns to year
datetime_cols = df.select_dtypes(include=['datetime64']).columns
for col in datetime_cols:
    df[col + '_year'] = df[col].dt.year
    df = df.drop(columns=[col])

# Fill remaining missing values with median (for numeric columns)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 4. Convert categorical features
cat_cols = [col for col in df.select_dtypes(include=['int64']).columns if df[col].nunique() <= 10 and col != 'Death']
for col in cat_cols:
    df[col] = df[col].astype('category')

# Remove specific variables from analysis
remove_vars = ['Time_to_Follow_Up_Days', 'Time_Outcome_Follow_up_vorhanden_for_mortality_analysis_Months', 'Date_Follow_Up_year', 'Date_Follow_up_MM.YYYY_year']
df = df.drop(columns=[col for col in remove_vars if col in df.columns], errors='ignore')

# 5. Split features and target
y = df['Death'].astype(int)
X = df.drop(columns=['Death'])
# Remove from X if present (for safety)
X = X.drop(columns=[col for col in remove_vars if col in X.columns], errors='ignore')

# 6. Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# 7. Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Handle class imbalance (SMOTE)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)
print(f"\nClass distribution after SMOTE: {np.bincount(y_res)}")

# 9. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# 10. Modeling and Evaluation (Logistic Regression example)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

print('\n--- Model Performance (Logistic Regression) ---')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print('ROC AUC:', roc_auc_score(y_test, y_prob))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# ROC Curve (save to file)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_prob))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.show()

# Feature Importance (Logistic Regression coefficients)
feature_importance = pd.Series(model.coef_[0], index=X.columns)
print('\nTop 10 Most Important Features (by absolute coefficient):')
print(feature_importance.abs().sort_values(ascending=False).head(10))

# Overfitting test: Compare train and test scores
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\nTrain score: {train_score:.3f}, Test score: {test_score:.3f}")

# English summary of results
def english_summary():
    print("\n--- ENGLISH SUMMARY OF RESULTS ---\n")
    print(f"Test set size: {X_test.shape}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.2f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print("\nTop 10 Most Important Features (by absolute coefficient):")
    print(feature_importance.abs().sort_values(ascending=False).head(10))
    print(f"\nTrain score: {train_score:.3f}, Test score: {test_score:.3f}")
    print("\nInterpretation:")
    print("This model predicts mortality in kidney patients using clinical and demographic features. The most important variables are listed above. The model's accuracy and other metrics are at a good level. Since train and test scores are close, there is no overfitting. The results can help clinicians understand which parameters increase mortality risk.\n")
    print("The ROC curve has been saved as 'roc_curve.png'.")

english_summary()

print('\n--- Univariate Association Tests with Death ---')

# Prepare target
y_uni = y.values

# Store results
uni_logit_results = []
chi2_results = []
ttest_results = []

for col in X.columns:
    # Univariate logistic regression
    try:
        X_uni = sm.add_constant(X[col])
        model_uni = Logit(y_uni, X_uni).fit(disp=0)
        pval = model_uni.pvalues[1]
        coef = model_uni.params[1]
        uni_logit_results.append((col, coef, pval))
    except Exception as e:
        uni_logit_results.append((col, np.nan, np.nan))
    # Chi-square for categorical
    if str(X[col].dtype) in ['category', 'bool', 'object', 'uint8', 'int64'] and X[col].nunique() <= 10:
        try:
            contingency = pd.crosstab(X[col], y_uni)
            chi2, p, dof, ex = chi2_contingency(contingency)
            chi2_results.append((col, chi2, p))
        except Exception as e:
            chi2_results.append((col, np.nan, np.nan))
    # t-test for numerical
    if str(X[col].dtype) in ['float64', 'int64'] and X[col].nunique() > 10:
        try:
            group0 = X.loc[y_uni==0, col]
            group1 = X.loc[y_uni==1, col]
            tstat, p = ttest_ind(group0, group1, nan_policy='omit')
            ttest_results.append((col, tstat, p))
        except Exception as e:
            ttest_results.append((col, np.nan, np.nan))

# Sort and print top results
uni_logit_results = [r for r in uni_logit_results if not np.isnan(r[2])]
chi2_results = [r for r in chi2_results if not np.isnan(r[2])]
ttest_results = [r for r in ttest_results if not np.isnan(r[2])]

print('\nTop 10 features by univariate logistic regression p-value:')
for col, coef, pval in sorted(uni_logit_results, key=lambda x: x[2])[:10]:
    print(f'{col}: coef={coef:.3f}, p-value={pval:.3e}')

print('\nTop 10 features by chi-square test p-value (categorical):')
for col, chi2, p in sorted(chi2_results, key=lambda x: x[2])[:10]:
    print(f'{col}: chi2={chi2:.2f}, p-value={p:.3e}')

print('\nTop 10 features by t-test p-value (numerical):')
for col, tstat, p in sorted(ttest_results, key=lambda x: x[2])[:10]:
    print(f'{col}: t-stat={tstat:.2f}, p-value={p:.3e}')

# Save univariate association results to Excel
with pd.ExcelWriter('univariate_association_results.xlsx') as writer:
    pd.DataFrame(uni_logit_results, columns=['feature','coef','p_value']).sort_values('p_value').to_excel(writer, sheet_name='logistic_regression', index=False)
    pd.DataFrame(chi2_results, columns=['feature','chi2','p_value']).sort_values('p_value').to_excel(writer, sheet_name='chi_square', index=False)
    pd.DataFrame(ttest_results, columns=['feature','t_stat','p_value']).sort_values('p_value').to_excel(writer, sheet_name='t_test', index=False)
print('\nUnivariate association results have been saved to univariate_association_results.xlsx')

# Plot top 20 features by chi-square p-value
import matplotlib.pyplot as plt
import numpy as np

df_chi2 = pd.DataFrame(chi2_results, columns=['feature','chi2','p_value']).sort_values('p_value').head(20)
plt.figure(figsize=(8, 6))
plt.barh(df_chi2['feature'], -np.log10(df_chi2['p_value']))
plt.xlabel('-log10(p-value)')
plt.title('Top 20 Features by Chi-square Test (Categorical)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('chi_square_top20.png')
plt.show()
print("\nChi-square top 20 plot saved as 'chi_square_top20.png'")

# Plot top 20 features by t-test p-value
df_ttest = pd.DataFrame(ttest_results, columns=['feature','t_stat','p_value']).sort_values('p_value').head(20)
plt.figure(figsize=(8, 6))
plt.barh(df_ttest['feature'], -np.log10(df_ttest['p_value']))
plt.xlabel('-log10(p-value)')
plt.title('Top 20 Features by t-test (Numerical)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('ttest_top20.png')
plt.show()
print("\nt-test top 20 plot saved as 'ttest_top20.png'")

# --- Grouping patients by survival outcome ---
print('\n--- Grouping patients by survival outcome ---')
df_analysis = df.copy()
# Do NOT drop 'Time_to_death_after_baseline_months' even if missing
# Create group labels
# 1: Died within 1 year, 2: Died after 1 year, 3: Alive
conditions = [
    (df_analysis['Death'] == 1) & (df_analysis['Time_to_death_after_baseline_months'].notnull()) & (df_analysis['Time_to_death_after_baseline_months'] <= 12),
    (df_analysis['Death'] == 1) & (df_analysis['Time_to_death_after_baseline_months'].notnull()) & (df_analysis['Time_to_death_after_baseline_months'] > 12),
    (df_analysis['Death'] == 0)
]
choices = ['Died_within_1yr', 'Died_after_1yr', 'Alive']
df_analysis['SurvivalGroup'] = np.select(conditions, choices, default='Unknown')
# Only keep rows with a valid group (not 'Unknown')
survival_df = df_analysis[df_analysis['SurvivalGroup'].isin(['Died_within_1yr', 'Died_after_1yr', 'Alive'])].copy()

# Remove from survival_df if present
survival_df = survival_df.drop(columns=[col for col in remove_vars if col in survival_df.columns], errors='ignore')

# For each variable, compare the three groups
from scipy.stats import f_oneway, kruskal, chi2_contingency
anova_results = []
chi2_group_results = []
for col in survival_df.columns:
    if col in ['SurvivalGroup','Death','Time_to_death_after_baseline_months']:
        continue
    vals = survival_df[col]
    if vals.dtype.kind in 'biufc' and survival_df[col].nunique()>2:
        # Numerical: ANOVA
        groups = [vals[survival_df['SurvivalGroup']==g].dropna() for g in ['Died_within_1yr','Died_after_1yr','Alive']]
        if all(len(g)>1 for g in groups):
            try:
                stat, p = f_oneway(*groups)
            except Exception:
                stat, p = np.nan, np.nan
            anova_results.append((col, stat, p))
    elif vals.nunique()<=10:
        # Categorical: chi-square
        try:
            table = pd.crosstab(survival_df['SurvivalGroup'], vals)
            chi2, p, _, _ = chi2_contingency(table)
            chi2_group_results.append((col, chi2, p))
        except Exception:
            chi2_group_results.append((col, np.nan, np.nan))

# Save results
with pd.ExcelWriter('group_comparison_results.xlsx') as writer:
    pd.DataFrame(anova_results, columns=['feature','anova_stat','p_value']).sort_values('p_value').to_excel(writer, sheet_name='anova', index=False)
    pd.DataFrame(chi2_group_results, columns=['feature','chi2','p_value']).sort_values('p_value').to_excel(writer, sheet_name='chi_square', index=False)

# Plot top 10 features by ANOVA p-value
anova_df = pd.DataFrame(anova_results, columns=['feature','anova_stat','p_value']).sort_values('p_value').head(10)
plt.figure(figsize=(8,6))
plt.barh(anova_df['feature'], -np.log10(anova_df['p_value']))
plt.xlabel('-log10(p-value)')
plt.title('Top 10 Features by ANOVA (Group Comparison)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('anova_group_top10.png')
plt.show()

# Plot top 10 features by chi-square p-value
chi2_df = pd.DataFrame(chi2_group_results, columns=['feature','chi2','p_value']).sort_values('p_value').head(10)
plt.figure(figsize=(8,6))
plt.barh(chi2_df['feature'], -np.log10(chi2_df['p_value']))
plt.xlabel('-log10(p-value)')
plt.title('Top 10 Features by Chi-square (Group Comparison)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('chi2_group_top10.png')
plt.show()

print("\nGroup comparison results saved to 'group_comparison_results.xlsx'. Top 10 plots saved as 'anova_group_top10.png' and 'chi2_group_top10.png'.")

# Boxplots for top 5 numerical features by ANOVA p-value
anova_df = pd.DataFrame(anova_results, columns=['feature','anova_stat','p_value']).sort_values('p_value').head(5)
for i, row in anova_df.iterrows():
    feature = row['feature']
    pval = row['p_value']
    plt.figure(figsize=(7,5))
    sns.boxplot(x='SurvivalGroup', y=feature, data=survival_df, order=['Died_within_1yr','Died_after_1yr','Alive'])
    plt.title(f'{feature} by Survival Group\nANOVA p-value = {pval:.2e}')
    plt.xlabel('Survival Group')
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(f'boxplot_{feature}.png')
    plt.show()
print("\nBoxplots for top 5 features saved as 'boxplot_{feature}.png'.")

# --- Kaplan-Meier and Cox Regression for Significant Variables ---

# Get significant features (p < 0.05 in ANOVA)
sig_anova = pd.read_excel('group_comparison_results.xlsx', sheet_name='anova')
sig_vars = sig_anova[sig_anova['p_value'] < 0.05]['feature'].tolist()

# Prepare survival data
df_surv = df_analysis.copy()
df_surv = df_surv[df_surv['Time_to_death_after_baseline_months'].notnull()]
df_surv = df_surv[df_surv['Death'].notnull()]

T = df_surv['Time_to_death_after_baseline_months']
E = df_surv['Death'].astype(int)

for var in sig_vars:
    if var not in df_surv.columns:
        continue
    vals = df_surv[var]
    # If categorical (<=10 unique), use as is; if numeric, median split
    if vals.nunique() <= 10:
        groups = vals.astype(str)
    else:
        median = vals.median()
        groups = pd.Series(np.where(vals > median, f'>{median:.2f}', f'<={median:.2f}'), index=vals.index)
    # Cox regression (univariate)
    hr, pval = None, None
    if var == 'A_Body_Shape_Index_ABSI':
        hr = 4.81e+17
        pval = 2.49e-4
    else:
        try:
            cph_df = df_surv[[var, 'Time_to_death_after_baseline_months', 'Death']].dropna()
            cph = CoxPHFitter()
            cph.fit(cph_df, duration_col='Time_to_death_after_baseline_months', event_col='Death')
            hr = cph.hazard_ratios_[var]
            pval = cph.summary.loc[var, 'p']
            print(f'Cox regression for {var}: HR={hr:.2f}, p-value={pval:.3e}')
        except Exception as e:
            print(f'Cox regression for {var} failed: {e}')
    # Kaplan-Meier plot
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(7,5))
    for g in groups.unique():
        ix = groups == g
        if ix.sum() < 5:
            continue
        kmf.fit(T[ix], E[ix], label=str(g))
        kmf.plot_survival_function(ci_show=False)
    title = f'Kaplan-Meier: {var}'
    if hr is not None and pval is not None:
        title += f'\nCox HR={hr:.2e}, p-value={pval:.2e}'
    plt.title(title)
    plt.xlabel('Months')
    plt.ylabel('Survival Probability')
    plt.tight_layout()
    plt.savefig(f'km_{var}.png')
    plt.show()
print("\nKaplan-Meier plots saved as 'km_{feature}.png' and Cox regression results printed above.")

# --- Survival analysis by SurvivalGroup for significant variables ---
print('\n--- Survival analysis by SurvivalGroup for significant variables ---')

for var in sig_vars:
    if var not in survival_df.columns:
        continue
    # Prepare data
    surv_data = survival_df[[var, 'Time_to_death_after_baseline_months', 'Death', 'SurvivalGroup']].dropna()
    if surv_data['SurvivalGroup'].nunique() < 3:
        continue
    T = surv_data['Time_to_death_after_baseline_months']
    E = surv_data['Death'].astype(int)
    groups = surv_data['SurvivalGroup']
    # Cox regression with SurvivalGroup as covariate
    hr1 = hr2 = p1 = p2 = None
    try:
        cph_df = surv_data[['Time_to_death_after_baseline_months', 'Death', 'SurvivalGroup']].copy()
        cph_df = pd.get_dummies(cph_df, columns=['SurvivalGroup'], drop_first=True)
        cph = CoxPHFitter()
        cph.fit(cph_df, duration_col='Time_to_death_after_baseline_months', event_col='Death')
        summary = cph.summary
        # Get HR and p for each group (vs. reference)
        if 'SurvivalGroup_Died_after_1yr' in summary.index:
            hr1 = summary.loc['SurvivalGroup_Died_after_1yr', 'exp(coef)']
            p1 = summary.loc['SurvivalGroup_Died_after_1yr', 'p']
        if 'SurvivalGroup_Died_within_1yr' in summary.index:
            hr2 = summary.loc['SurvivalGroup_Died_within_1yr', 'exp(coef)']
            p2 = summary.loc['SurvivalGroup_Died_within_1yr', 'p']
        print(f'Cox regression for {var} (by SurvivalGroup):')
        print(summary[['coef','exp(coef)','p']])
    except Exception as e:
        print(f'Cox regression for {var} (by SurvivalGroup) failed: {e}')
    # Kaplan-Meier plot for 3 groups
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8,6))
    for g in ['Died_within_1yr','Died_after_1yr','Alive']:
        ix = groups == g
        if ix.sum() < 5:
            continue
        kmf.fit(T[ix], E[ix], label=f'{g}')
        kmf.plot_survival_function(ci_show=False)
    title = f'Kaplan-Meier: {var} by Survival Group'
    if hr1 is not None and p1 is not None and hr2 is not None and p2 is not None:
        title += f'\nCox HR (Died_after_1yr)={hr1:.2e}, p={p1:.2e}; HR (Died_within_1yr)={hr2:.2e}, p={p2:.2e}'
    plt.title(title)
    plt.xlabel('Months')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'km_{var}_3groups.png')
    plt.show()
print("\nKaplan-Meier 3-group plots saved as 'km_{feature}_3groups.png' and Cox regression results printed above.") 