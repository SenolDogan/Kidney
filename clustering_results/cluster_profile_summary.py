import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Veriyi yükle
file_path = '../kidney.xlsx' if __name__ == '__main__' else 'kidney.xlsx'
df = pd.read_excel('kidney.xlsx')

features = [
    'AGE_Baseline',
    'Sex_1male_2female',
    'Diabetes_status_1yes_0no',
    'eGFR_CKD_EPI_Creatinine_at_Baseline',
    'BMI'
]

for col in features:
    df[col] = df[col].fillna(df[col].median())

cat_cols = ['Sex_1male_2female', 'Diabetes_status_1yes_0no']
df[cat_cols] = df[cat_cols].astype(int)
df_encoded = pd.get_dummies(df[features], columns=cat_cols, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

labels = AgglomerativeClustering(n_clusters=3).fit_predict(X_scaled)
df['Cluster'] = labels

summary = df.groupby('Cluster')[['AGE_Baseline','eGFR_CKD_EPI_Creatinine_at_Baseline','BMI']].agg(['mean','median'])
summary['Female_ratio'] = df.groupby('Cluster')['Sex_1male_2female'].apply(lambda x: (x==2).mean())
summary['Diabetes_ratio'] = df.groupby('Cluster')['Diabetes_status_1yes_0no'].mean()
summary.to_excel('clustering_results/hierarchical_cluster_profile_summary.xlsx')
print(summary) 