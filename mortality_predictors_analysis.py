import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import os

# Set style for professional plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Read data
df = pd.read_excel('kidney.xlsx')

# Clean data
duration_col = 'Time_to_death_after_baseline_months'
event_col = 'Death'
df[duration_col] = pd.to_numeric(df[duration_col], errors='coerce')
df[event_col] = pd.to_numeric(df[event_col], errors='coerce')
df = df.dropna(subset=[duration_col, event_col])
df[event_col] = df[event_col].astype(int)

# Create plots directory
os.makedirs('mortality_plots', exist_ok=True)

# Define mortality predictors with correct column names from dataset
predictors = {
    'eGFR_CKD_EPI_Creatinine_at_Baseline': {'type': 'continuous', 'method': 'median', 'title': 'eGFR (CKD-EPI)'},
    'A_Body_Shape_Index_ABSI': {'type': 'continuous', 'method': 'median', 'title': 'ABSI'},
    'WWI_Weight_adjusted_Waist_Index': {'type': 'continuous', 'method': 'median', 'title': 'WWI'},
    'ConI_Conicity_Index': {'type': 'continuous', 'method': 'median', 'title': 'ConI'},
    'BRI_Body_Roundness_Index': {'type': 'continuous', 'method': 'median', 'title': 'BRI'},
    'AGE_Baseline': {'type': 'continuous', 'method': 'median', 'title': 'Age'},
    'Body_Fat_Percentage': {'type': 'continuous', 'method': 'median', 'title': 'Body Fat (%)'},
    'Urinary_Albumin_Creatinine_ratio_mg_g': {'type': 'continuous', 'method': 'median', 'title': 'Albuminuria (mg/g)'},
    'Diabetes_status_1yes_0no': {'type': 'categorical', 'method': 'direct', 'title': 'Diabetes'}
}

def create_km_plot(var_name, var_config):
    """Create Kaplan-Meier plot for a specific variable"""
    
    if var_name not in df.columns:
        print(f"Warning: {var_name} not found in dataset")
        return
    
    plt.figure(figsize=(10, 6))
    
    if var_config['type'] == 'categorical':
        # Categorical variable
        groups = df[var_name].unique()
        if len(groups) != 2:
            print(f"Warning: {var_name} has {len(groups)} groups, skipping")
            plt.close()
            return
            
        mask1 = df[var_name] == groups[0]
        mask2 = df[var_name] == groups[1]
        label1 = f"{var_config['title']} = {groups[0]}"
        label2 = f"{var_config['title']} = {groups[1]}"
        
    else:
        # Continuous variable - split by median
        median_val = df[var_name].median()
        mask1 = df[var_name] <= median_val
        mask2 = df[var_name] > median_val
        label1 = f"{var_config['title']} ≤ {median_val:.2f}"
        label2 = f"{var_config['title']} > {median_val:.2f}"
    
    # Create KM plots
    kmf = KaplanMeierFitter()
    
    # Group 1
    kmf.fit(df[mask1][duration_col], event_observed=df[mask1][event_col], label=label1)
    kmf.plot_survival_function(color='blue', linewidth=2)
    
    # Group 2
    kmf.fit(df[mask2][duration_col], event_observed=df[mask2][event_col], label=label2)
    kmf.plot_survival_function(color='red', linewidth=2)
    
    # Calculate log-rank p-value
    try:
        results = logrank_test(
            df[mask1][duration_col], df[mask2][duration_col],
            event_observed_A=df[mask1][event_col], event_observed_B=df[mask2][event_col]
        )
        p_value = results.p_value
        p_text = f"Log-rank p-value: {p_value:.4f}"
        if p_value < 0.001:
            p_text = f"Log-rank p-value: < 0.001"
        elif p_value < 0.01:
            p_text = f"Log-rank p-value: {p_value:.3f} **"
        elif p_value < 0.05:
            p_text = f"Log-rank p-value: {p_value:.3f} *"
    except Exception as e:
        p_text = "Log-rank p-value: N/A"
        print(f"Error calculating p-value for {var_name}: {e}")
    
    # Customize plot
    plt.title(f"Kaplan-Meier Survival Analysis: {var_config['title']}", fontsize=16, fontweight='bold')
    plt.xlabel("Time (months)", fontsize=14)
    plt.ylabel("Survival Probability", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper right')
    
    # Add p-value text
    plt.text(0.02, 0.98, p_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    safe_title = var_config['title'].replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('/', '_').replace('-', '_')
    filename = f"mortality_plots/KM_{safe_title}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# Create plots for each predictor
print("Creating mortality predictor plots...")
for var_name, var_config in predictors.items():
    create_km_plot(var_name, var_config)

print("\nAll plots have been created and saved in 'mortality_plots' folder!")
print("Files created:")
for var_name, var_config in predictors.items():
    safe_title = var_config['title'].replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('/', '_').replace('-', '_')
    filename = f"mortality_plots/KM_{safe_title}.png"
    if os.path.exists(filename):
        print(f"✓ {filename}")
    else:
        print(f"✗ {filename} (not created)") 