import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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

# Remove rows where Death is NaN
df = df.dropna(subset=[event_col])
df[event_col] = df[event_col].astype(int)

# Create three groups based on survival time and death status
def create_three_groups(row):
    if row[event_col] == 0:  # Alive
        return 'Alive'
    elif row[duration_col] <= 12:  # Died within 1 year (12 months)
        return 'Died ≤ 1 year'
    else:  # Died after more than 1 year
        return 'Died > 1 year'

# For alive patients, we need to create a follow-up time
# Let's use the maximum follow-up time from the dataset as a reference
max_followup = df[duration_col].max() if not df[duration_col].isna().all() else 60

# Create a new column for analysis that includes follow-up time for alive patients
df['Analysis_Time'] = df.apply(lambda row: 
    max_followup if row[event_col] == 0 else row[duration_col], axis=1)

df['Three_Groups'] = df.apply(create_three_groups, axis=1)

# Create plots directory
os.makedirs('mortality_boxplots', exist_ok=True)

# Define mortality predictors with correct column names from dataset
predictors = {
    'eGFR_CKD_EPI_Creatinine_at_Baseline': {'type': 'continuous', 'title': 'eGFR (CKD-EPI)'},
    'A_Body_Shape_Index_ABSI': {'type': 'continuous', 'title': 'ABSI'},
    'WWI_Weight_adjusted_Waist_Index': {'type': 'continuous', 'title': 'WWI'},
    'ConI_Conicity_Index': {'type': 'continuous', 'title': 'ConI'},
    'BRI_Body_Roundness_Index': {'type': 'continuous', 'title': 'BRI'},
    'AGE_Baseline': {'type': 'continuous', 'title': 'Age'},
    'Body_Fat_Percentage': {'type': 'continuous', 'title': 'Body Fat (%)'},
    'Urinary_Albumin_Creatinine_ratio_mg_g': {'type': 'continuous', 'title': 'Albuminuria (mg/g)'},
    'Diabetes_status_1yes_0no': {'type': 'categorical', 'title': 'Diabetes'}
}

def create_boxplot_with_pvalues(var_name, var_config):
    """Create box plot for three groups with p-values"""
    
    if var_name not in df.columns:
        print(f"Warning: {var_name} not found in dataset")
        return
    
    plt.figure(figsize=(12, 8))
    
    if var_config['type'] == 'categorical':
        # For categorical variables, create box plot of the variable itself
        # Group by the categorical variable and show distribution across three groups
        data_for_plot = []
        labels_for_plot = []
        
        for cat_value in df[var_name].unique():
            cat_mask = df[var_name] == cat_value
            cat_data = df[cat_mask]
            
            for group_name in ['Alive', 'Died ≤ 1 year', 'Died > 1 year']:
                group_mask = cat_data['Three_Groups'] == group_name
                if group_mask.sum() > 0:
                    data_for_plot.append(cat_data[group_mask][var_name].values)
                    labels_for_plot.append(f"{var_config['title']} = {cat_value} - {group_name}")
        
        if len(data_for_plot) > 0:
            bp = plt.boxplot(data_for_plot, tick_labels=labels_for_plot, patch_artist=True)
            
            # Color the boxes
            colors = ['lightgreen', 'orange', 'lightcoral']
            for i, patch in enumerate(bp['boxes']):
                if i < len(colors):
                    patch.set_facecolor(colors[i % len(colors)])
            
    else:
        # For continuous variables, create box plot of the variable values across three groups
        data_for_plot = []
        labels_for_plot = []
        groups_with_data = []
        
        for group_name in ['Alive', 'Died ≤ 1 year', 'Died > 1 year']:
            group_mask = df['Three_Groups'] == group_name
            if group_mask.sum() > 0:
                group_data = df[group_mask][var_name].dropna().values
                if len(group_data) > 0:
                    data_for_plot.append(group_data)
                    labels_for_plot.append(group_name)
                    groups_with_data.append(group_name)
        
        if len(data_for_plot) > 0:
            bp = plt.boxplot(data_for_plot, tick_labels=labels_for_plot, patch_artist=True)
            
            # Color the boxes with distinct colors
            colors = ['lightgreen', 'orange', 'lightcoral']
            for i, patch in enumerate(bp['boxes']):
                if i < len(colors):
                    patch.set_facecolor(colors[i])
                    patch.set_alpha(0.7)
    
    # Calculate p-values between groups
    p_values = []
    
    if var_config['type'] == 'continuous':
        # For continuous variables, compare groups that have data
        groups_data = {}
        for group_name in ['Alive', 'Died ≤ 1 year', 'Died > 1 year']:
            group_mask = df['Three_Groups'] == group_name
            if group_mask.sum() > 0:
                group_data = df[group_mask][var_name].dropna().values
                if len(group_data) > 0:
                    groups_data[group_name] = group_data
        
        if len(groups_data) >= 2:
            group_names = list(groups_data.keys())
            for i in range(len(group_names)):
                for j in range(i+1, len(group_names)):
                    if len(groups_data[group_names[i]]) > 0 and len(groups_data[group_names[j]]) > 0:
                        try:
                            # Use Mann-Whitney U test for non-parametric comparison
                            stat, p_val = stats.mannwhitneyu(groups_data[group_names[i]], 
                                                           groups_data[group_names[j]], 
                                                           alternative='two-sided')
                            p_values.append((f"{group_names[i]} vs {group_names[j]}", p_val))
                        except Exception as e:
                            print(f"Error calculating p-value for {group_names[i]} vs {group_names[j]}: {e}")
                            p_values.append((f"{group_names[i]} vs {group_names[j]}", None))
    
    # Customize plot
    plt.title(f"Box Plot Analysis: {var_config['title']} by Mortality Groups", fontsize=16, fontweight='bold')
    plt.ylabel(var_config['title'], fontsize=14)
    plt.xlabel("Groups", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add p-values to the plot with better formatting
    if p_values:
        p_text = "P-values (Mann-Whitney U test):\n"
        for group_comparison, p_val in p_values:
            if p_val is not None:
                if p_val < 0.001:
                    p_text += f"{group_comparison}: < 0.001 ***\n"
                elif p_val < 0.01:
                    p_text += f"{group_comparison}: {p_val:.3f} **\n"
                elif p_val < 0.05:
                    p_text += f"{group_comparison}: {p_val:.3f} *\n"
                else:
                    p_text += f"{group_comparison}: {p_val:.3f}\n"
            else:
                p_text += f"{group_comparison}: N/A\n"
        
        # Add p-value text box with better positioning
        plt.text(0.02, 0.98, p_text, transform=plt.gca().transAxes, 
                 fontsize=11, verticalalignment='top', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black'))
    
    # Add sample sizes to the plot
    if var_config['type'] == 'continuous':
        sample_size_text = "Sample sizes:\n"
        for group_name in ['Alive', 'Died ≤ 1 year', 'Died > 1 year']:
            group_mask = df['Three_Groups'] == group_name
            if group_mask.sum() > 0:
                group_data = df[group_mask][var_name].dropna()
                if len(group_data) > 0:
                    sample_size_text += f"{group_name}: n={len(group_data)}\n"
        
        # Add sample size text to the right side
        plt.text(0.98, 0.98, sample_size_text, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    safe_title = var_config['title'].replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('/', '_').replace('-', '_')
    filename = f"mortality_boxplots/Boxplot_{safe_title}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# Create box plots for each predictor
print("Creating mortality predictor box plots...")
for var_name, var_config in predictors.items():
    create_boxplot_with_pvalues(var_name, var_config)

print("\nAll box plots have been created and saved in 'mortality_boxplots' folder!")
print("Files created:")
for var_name, var_config in predictors.items():
    safe_title = var_config['title'].replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('/', '_').replace('-', '_')
    filename = f"mortality_boxplots/Boxplot_{safe_title}.png"
    if os.path.exists(filename):
        print(f"✓ {filename}")
    else:
        print(f"✗ {filename} (not created)")

# Print summary of three groups
print(f"\nSummary of Three Groups:")
print(df['Three_Groups'].value_counts())

# Print summary statistics for each group
print(f"\nSummary Statistics by Groups:")
for var_name, var_config in predictors.items():
    if var_name in df.columns and var_config['type'] == 'continuous':
        print(f"\n{var_config['title']}:")
        for group_name in ['Alive', 'Died ≤ 1 year', 'Died > 1 year']:
            group_mask = df['Three_Groups'] == group_name
            if group_mask.sum() > 0:
                group_data = df[group_mask][var_name].dropna()
                if len(group_data) > 0:
                    print(f"  {group_name}: n={len(group_data)}, mean={group_data.mean():.2f}, median={group_data.median():.2f}")
                else:
                    print(f"  {group_name}: n=0 (no data)")
            else:
                print(f"  {group_name}: n=0 (no patients)") 