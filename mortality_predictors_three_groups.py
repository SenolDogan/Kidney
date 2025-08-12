import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import os

# Set style for professional plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
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

# Create three groups based on survival time and death status
def create_three_groups(row):
    if row[event_col] == 0:  # Alive
        return 'Alive'
    elif row[duration_col] <= 12:  # Died within 1 year (12 months)
        return 'Died ≤ 1 year'
    else:  # Died after more than 1 year
        return 'Died > 1 year'

df['Three_Groups'] = df.apply(create_three_groups, axis=1)

# Create plots directory
os.makedirs('mortality_three_groups_plots', exist_ok=True)

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

def create_three_groups_km_plot(var_name, var_config):
    """Create Kaplan-Meier plot for three groups based on a specific variable"""
    
    if var_name not in df.columns:
        print(f"Warning: {var_name} not found in dataset")
        return
    
    plt.figure(figsize=(12, 8))
    
    if var_config['type'] == 'categorical':
        # Categorical variable
        groups = df[var_name].unique()
        if len(groups) != 2:
            print(f"Warning: {var_name} has {len(groups)} groups, skipping")
            plt.close()
            return
            
        # Create masks for each category
        cat1_mask = df[var_name] == groups[0]
        cat2_mask = df[var_name] == groups[1]
        
        # For each category, create three groups
        for cat_idx, (cat_mask, cat_label) in enumerate([(cat1_mask, f"{var_config['title']} = {groups[0]}"), 
                                                         (cat2_mask, f"{var_config['title']} = {groups[1]}")]):
            
            cat_data = df[cat_mask]
            if len(cat_data) == 0:
                continue
                
            # Create three groups for this category
            cat_data['Three_Groups_Cat'] = cat_data.apply(create_three_groups, axis=1)
            
            # Plot KM for each group
            colors = ['green', 'orange', 'red']
            for group_idx, group_name in enumerate(['Alive', 'Died ≤ 1 year', 'Died > 1 year']):
                group_mask = cat_data['Three_Groups_Cat'] == group_name
                if group_mask.sum() > 0:
                    kmf = KaplanMeierFitter()
                    kmf.fit(cat_data[group_mask][duration_col], 
                           event_observed=cat_data[group_mask][event_col], 
                           label=f"{cat_label} - {group_name}")
                    kmf.plot_survival_function(color=colors[group_idx], linewidth=2, linestyle='-' if cat_idx == 0 else '--')
        
    else:
        # Continuous variable - split by median
        median_val = df[var_name].median()
        low_mask = df[var_name] <= median_val
        high_mask = df[var_name] > median_val
        
        # For each group (low/high), create three groups
        for group_idx, (group_mask, group_label) in enumerate([(low_mask, f"{var_config['title']} ≤ {median_val:.2f}"), 
                                                               (high_mask, f"{var_config['title']} > {median_val:.2f}")]):
            
            group_data = df[group_mask]
            if len(group_data) == 0:
                continue
                
            # Create three groups for this continuous group
            group_data['Three_Groups_Cont'] = group_data.apply(create_three_groups, axis=1)
            
            # Plot KM for each group
            colors = ['green', 'orange', 'red']
            for three_group_idx, group_name in enumerate(['Alive', 'Died ≤ 1 year', 'Died > 1 year']):
                three_group_mask = group_data['Three_Groups_Cont'] == group_name
                if three_group_mask.sum() > 0:
                    kmf = KaplanMeierFitter()
                    kmf.fit(group_data[three_group_mask][duration_col], 
                           event_observed=group_data[three_group_mask][event_col], 
                           label=f"{group_label} - {group_name}")
                    kmf.plot_survival_function(color=colors[three_group_idx], linewidth=2, linestyle='-' if group_idx == 0 else '--')
    
    # Calculate overall log-rank p-value for three groups
    try:
        # Create overall three groups for the entire dataset
        overall_three_groups = df['Three_Groups'].value_counts()
        if len(overall_three_groups) >= 2:
            # Use the two largest groups for log-rank test
            largest_groups = overall_three_groups.nlargest(2).index
            group1_mask = df['Three_Groups'] == largest_groups[0]
            group2_mask = df['Three_Groups'] == largest_groups[1]
            
            results = logrank_test(
                df[group1_mask][duration_col], df[group2_mask][duration_col],
                event_observed=df[group1_mask][event_col], event_observed_B=df[group2_mask][event_col]
            )
            p_value = results.p_value
            p_text = f"Overall Log-rank p-value: {p_value:.4f}"
            if p_value < 0.001:
                p_text = f"Overall Log-rank p-value: < 0.001"
            elif p_value < 0.01:
                p_text = f"Overall Log-rank p-value: {p_value:.3f} **"
            elif p_value < 0.05:
                p_text = f"Overall Log-rank p-value: {p_value:.3f} *"
        else:
            p_text = "Overall Log-rank p-value: N/A"
    except Exception as e:
        p_text = "Overall Log-rank p-value: N/A"
        print(f"Error calculating overall p-value for {var_name}: {e}")
    
    # Customize plot
    plt.title(f"Three-Group Survival Analysis: {var_config['title']}", fontsize=16, fontweight='bold')
    plt.xlabel("Time (months)", fontsize=14)
    plt.ylabel("Survival Probability", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add p-value text
    plt.text(0.02, 0.98, p_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    safe_title = var_config['title'].replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('/', '_').replace('-', '_')
    filename = f"mortality_three_groups_plots/ThreeGroups_{safe_title}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# Create plots for each predictor
print("Creating three-group mortality predictor plots...")
for var_name, var_config in predictors.items():
    create_three_groups_km_plot(var_name, var_config)

print("\nAll three-group plots have been created and saved in 'mortality_three_groups_plots' folder!")
print("Files created:")
for var_name, var_config in predictors.items():
    safe_title = var_config['title'].replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('/', '_').replace('-', '_')
    filename = f"mortality_three_groups_plots/ThreeGroups_{safe_title}.png"
    if os.path.exists(filename):
        print(f"✓ {filename}")
    else:
        print(f"✗ {filename} (not created)")

# Print summary of three groups
print(f"\nSummary of Three Groups:")
print(df['Three_Groups'].value_counts()) 