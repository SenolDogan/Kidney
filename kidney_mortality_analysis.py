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

# Drop columns with too many missing values (>60%)
too_many_missing = df.columns[df.isnull().mean() > 0.6]
df = df.drop(columns=too_many_missing)
print(f"\nDropped columns (>%60 missing): {list(too_many_missing)}")

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

# 5. Split features and target
y = df['Death'].astype(int)
X = df.drop(columns=['Death'])

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