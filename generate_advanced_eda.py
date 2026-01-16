import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create directories
os.makedirs('figures', exist_ok=True)
os.makedirs('reports', exist_ok=True)

print("Loading data...")
# Load data
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

# Remove first unnamed column if exists
if train_df.columns[0] == 'Unnamed: 0':
    train_df = train_df.drop(columns=['Unnamed: 0'])
if test_df.columns[0] == 'Unnamed: 0':
    test_df = test_df.drop(columns=['Unnamed: 0'])

print(f"Training set: {train_df.shape}")
print(f"Test set: {test_df.shape}")

# Combine for some analyses (excluding target from test)
test_df_no_target = test_df.copy()
if 'satisfaction' in test_df.columns:
    test_df_no_target = test_df.drop(columns=['satisfaction'])
combined_df = pd.concat([train_df, test_df_no_target], ignore_index=True)

print("\n" + "="*80)
print("ADVANCED EDA - ADDITIONAL ANALYSES")
print("="*80)

# ============================================================================
# 1. Target variable analysis by different segments
# ============================================================================
print("\n1. Target variable analysis by different segments...")

# Encode target for analysis
train_df['satisfaction_encoded'] = train_df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})

# 1.1 Satisfaction by gender
print("\n1.1 Satisfaction by gender:")
gender_satisfaction = train_df.groupby('Gender')['satisfaction_encoded'].mean() * 100
print(gender_satisfaction)

# 1.2 Satisfaction by customer type
print("\n1.2 Satisfaction by customer type:")
customer_satisfaction = train_df.groupby('Customer Type')['satisfaction_encoded'].mean() * 100
print(customer_satisfaction)

# 1.3 Satisfaction by travel type
print("\n1.3 Satisfaction by travel type:")
travel_satisfaction = train_df.groupby('Type of Travel')['satisfaction_encoded'].mean() * 100
print(travel_satisfaction)

# 1.4 Satisfaction by class
print("\n1.4 Satisfaction by class:")
class_satisfaction = train_df.groupby('Class')['satisfaction_encoded'].mean() * 100
print(class_satisfaction)

# ============================================================================
# 2. Create visualization: Satisfaction by different segments
# ============================================================================
print("\n2. Creating satisfaction by segments visualization...")

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Satisfaction Rate by Different Segments', fontsize=16)

# Gender
gender_counts = train_df.groupby(['Gender', 'satisfaction']).size().unstack()
gender_counts.plot(kind='bar', stacked=True, ax=axes1[0, 0], color=['#ff7f0e', '#1f77b4'])
axes1[0, 0].set_title('Satisfaction by Gender')
axes1[0, 0].set_xlabel('Gender')
axes1[0, 0].set_ylabel('Count')
axes1[0, 0].legend(title='Satisfaction')
axes1[0, 0].tick_params(axis='x', rotation=0)

# Customer Type
customer_counts = train_df.groupby(['Customer Type', 'satisfaction']).size().unstack()
customer_counts.plot(kind='bar', stacked=True, ax=axes1[0, 1], color=['#ff7f0e', '#1f77b4'])
axes1[0, 1].set_title('Satisfaction by Customer Type')
axes1[0, 1].set_xlabel('Customer Type')
axes1[0, 1].set_ylabel('Count')
axes1[0, 1].legend(title='Satisfaction')
axes1[0, 1].tick_params(axis='x', rotation=0)

# Type of Travel
travel_counts = train_df.groupby(['Type of Travel', 'satisfaction']).size().unstack()
travel_counts.plot(kind='bar', stacked=True, ax=axes1[1, 0], color=['#ff7f0e', '#1f77b4'])
axes1[1, 0].set_title('Satisfaction by Type of Travel')
axes1[1, 0].set_xlabel('Type of Travel')
axes1[1, 0].set_ylabel('Count')
axes1[1, 0].legend(title='Satisfaction')
axes1[1, 0].tick_params(axis='x', rotation=0)

# Class
class_counts = train_df.groupby(['Class', 'satisfaction']).size().unstack()
class_counts.plot(kind='bar', stacked=True, ax=axes1[1, 1], color=['#ff7f0e', '#1f77b4'])
axes1[1, 1].set_title('Satisfaction by Class')
axes1[1, 1].set_xlabel('Class')
axes1[1, 1].set_ylabel('Count')
axes1[1, 1].legend(title='Satisfaction')
axes1[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('figures/satisfaction_by_segments.png', dpi=150, bbox_inches='tight')
print("Saved: figures/satisfaction_by_segments.png")
plt.close(fig1)

# ============================================================================
# 3. Age analysis by satisfaction
# ============================================================================
print("\n3. Age analysis by satisfaction...")

# 3.1 Age distribution by satisfaction
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
satisfied_ages = train_df[train_df['satisfaction'] == 'satisfied']['Age']
neutral_ages = train_df[train_df['satisfaction'] == 'neutral or dissatisfied']['Age']

axes2[0].hist([satisfied_ages, neutral_ages], bins=30, label=['Satisfied', 'Neutral/Dissatisfied'],
              color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black')
axes2[0].set_xlabel('Age')
axes2[0].set_ylabel('Frequency')
axes2[0].set_title('Age Distribution by Satisfaction')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

# Box plot
sns.boxplot(x='satisfaction', y='Age', data=train_df, ax=axes2[1], hue='satisfaction', palette=['#ff7f0e', '#1f77b4'], legend=False)
axes2[1].set_xlabel('Satisfaction')
axes2[1].set_ylabel('Age')
axes2[1].set_title('Age Distribution by Satisfaction (Box Plot)')

plt.tight_layout()
plt.savefig('figures/age_by_satisfaction.png', dpi=150, bbox_inches='tight')
print("Saved: figures/age_by_satisfaction.png")
plt.close(fig2)

# 3.2 Age groups analysis
print("\n3.2 Age groups analysis...")

# Create age groups
def create_age_groups(age):
    if age <= 20:
        return 'Teen (≤20)'
    elif age <= 30:
        return '20s'
    elif age <= 40:
        return '30s'
    elif age <= 50:
        return '40s'
    elif age <= 60:
        return '50s'
    else:
        return '60+'

train_df['age_group'] = train_df['Age'].apply(create_age_groups)

# Calculate satisfaction rate by age group
age_group_satisfaction = train_df.groupby('age_group')['satisfaction_encoded'].mean() * 100
age_group_counts = train_df['age_group'].value_counts()

print("\nSatisfaction rate by age group:")
for age_group in age_group_satisfaction.index:
    rate = age_group_satisfaction[age_group]
    count = age_group_counts[age_group]
    print(f"  {age_group}: {rate:.1f}% satisfied (n={count})")

# ============================================================================
# 4. Service ratings analysis by satisfaction
# ============================================================================
print("\n4. Service ratings analysis by satisfaction...")

# List of service rating columns
service_columns = [
    'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Inflight service', 'Cleanliness'
]

# Calculate mean ratings by satisfaction
satisfied_means = train_df[train_df['satisfaction'] == 'satisfied'][service_columns].mean()
neutral_means = train_df[train_df['satisfaction'] == 'neutral or dissatisfied'][service_columns].mean()

# Create comparison dataframe
rating_comparison = pd.DataFrame({
    'Satisfied': satisfied_means,
    'Neutral/Dissatisfied': neutral_means,
    'Difference': satisfied_means - neutral_means
}).sort_values('Difference', ascending=False)

print("\nTop service differences (Satisfied - Neutral/Dissatisfied):")
print(rating_comparison.head(10))

# Save to CSV
rating_comparison.to_csv('reports/service_rating_differences.csv')
print("Saved service rating differences to: reports/service_rating_differences.csv")

# 4.1 Visualization: Service rating differences
fig3, ax3 = plt.subplots(figsize=(12, 8))
x = np.arange(len(rating_comparison))
width = 0.35

bars1 = ax3.barh(x - width/2, rating_comparison['Satisfied'], width, label='Satisfied', color='#1f77b4')
bars2 = ax3.barh(x + width/2, rating_comparison['Neutral/Dissatisfied'], width, label='Neutral/Dissatisfied', color='#ff7f0e')

ax3.set_yticks(x)
ax3.set_yticklabels(rating_comparison.index)
ax3.set_xlabel('Average Rating (0-5)')
ax3.set_title('Service Ratings by Satisfaction Level')
ax3.legend()
ax3.invert_yaxis()  # Highest difference at top

plt.tight_layout()
plt.savefig('figures/service_ratings_by_satisfaction.png', dpi=150, bbox_inches='tight')
print("Saved: figures/service_ratings_by_satisfaction.png")
plt.close(fig3)

# ============================================================================
# 5. Delay analysis by satisfaction
# ============================================================================
print("\n5. Delay analysis by satisfaction...")

# 5.1 Delay distributions
fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))

# Departure delay
sns.boxplot(x='satisfaction', y='Departure Delay in Minutes', data=train_df, ax=axes4[0], hue='satisfaction', palette=['#ff7f0e', '#1f77b4'], legend=False)
axes4[0].set_xlabel('Satisfaction')
axes4[0].set_ylabel('Departure Delay (minutes)')
axes4[0].set_title('Departure Delay by Satisfaction')
axes4[0].set_yscale('log')  # Log scale due to outliers

# Arrival delay
sns.boxplot(x='satisfaction', y='Arrival Delay in Minutes', data=train_df, ax=axes4[1], hue='satisfaction', palette=['#ff7f0e', '#1f77b4'], legend=False)
axes4[1].set_xlabel('Satisfaction')
axes4[1].set_ylabel('Arrival Delay (minutes)')
axes4[1].set_title('Arrival Delay by Satisfaction')
axes4[1].set_yscale('log')  # Log scale due to outliers

plt.tight_layout()
plt.savefig('figures/delays_by_satisfaction.png', dpi=150, bbox_inches='tight')
print("Saved: figures/delays_by_satisfaction.png")
plt.close(fig4)

# 5.2 Delay statistics
print("\nDelay statistics by satisfaction:")
delay_stats = train_df.groupby('satisfaction')[['Departure Delay in Minutes', 'Arrival Delay in Minutes']].agg(['mean', 'median', 'std'])
print(delay_stats)

# ============================================================================
# 6. Interaction analysis: Satisfaction by Class and Type of Travel
# ============================================================================
print("\n6. Interaction analysis: Satisfaction by Class and Type of Travel...")

# Create pivot table
pivot_table = train_df.pivot_table(
    index='Class',
    columns='Type of Travel',
    values='satisfaction_encoded',
    aggfunc='mean'
) * 100

print("\nSatisfaction rate by Class and Type of Travel (%):")
print(pivot_table)

# 6.1 Visualization: Heatmap of interactions
fig5, ax5 = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Satisfaction Rate (%)'})
ax5.set_title('Satisfaction Rate by Class and Type of Travel', fontsize=14)
ax5.set_xlabel('Type of Travel', fontsize=12)
ax5.set_ylabel('Class', fontsize=12)

plt.tight_layout()
plt.savefig('figures/satisfaction_class_travel_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: figures/satisfaction_class_travel_heatmap.png")
plt.close(fig5)

# ============================================================================
# 7. Correlation with target
# ============================================================================
print("\n7. Correlation with target...")

# Prepare data for correlation
corr_df = train_df.copy()

# Encode categorical variables for correlation
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'age_group']:
    if col in corr_df.columns:
        corr_df[col] = pd.factorize(corr_df[col])[0]

# Drop original satisfaction column (string) and other non-numeric columns
columns_to_drop = ['satisfaction']
if 'distance_category' in corr_df.columns:
    columns_to_drop.append('distance_category')
corr_df = corr_df.drop(columns=columns_to_drop)

# Calculate correlations with target
correlations = corr_df.corr()['satisfaction_encoded'].drop('satisfaction_encoded').sort_values(ascending=False)

print("\nTop 10 features correlated with satisfaction (positive):")
print(correlations.head(10))

print("\nTop 10 features correlated with satisfaction (negative):")
print(correlations.tail(10))

# Save correlations to CSV
correlations.to_csv('reports/correlations_with_target.csv')
print("Saved correlations with target to: reports/correlations_with_target.csv")

# 7.1 Visualization: Top correlations with target
fig6, ax6 = plt.subplots(figsize=(10, 8))
top_correlations = pd.concat([correlations.head(15), correlations.tail(5)])
colors = ['green' if x > 0 else 'red' for x in top_correlations]

top_correlations.plot(kind='barh', color=colors, ax=ax6)
ax6.set_xlabel('Correlation with Satisfaction')
ax6.set_title('Top Features Correlated with Satisfaction', fontsize=14)
ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax6.invert_yaxis()

plt.tight_layout()
plt.savefig('figures/correlations_with_target.png', dpi=150, bbox_inches='tight')
print("Saved: figures/correlations_with_target.png")
plt.close(fig6)

# ============================================================================
# 8. Flight distance analysis
# ============================================================================
print("\n8. Flight distance analysis...")

# 8.1 Flight distance categories
def create_distance_categories(distance):
    if distance <= 500:
        return 'Short (≤500 mi)'
    elif distance <= 1500:
        return 'Medium (501-1500 mi)'
    else:
        return 'Long (>1500 mi)'

train_df['distance_category'] = train_df['Flight Distance'].apply(create_distance_categories)

# Calculate satisfaction by distance category
distance_satisfaction = train_df.groupby('distance_category')['satisfaction_encoded'].mean() * 100
distance_counts = train_df['distance_category'].value_counts()

print("\nSatisfaction by flight distance category:")
for category in distance_satisfaction.index:
    rate = distance_satisfaction[category]
    count = distance_counts[category]
    print(f"  {category}: {rate:.1f}% satisfied (n={count})")

# 8.2 Visualization
fig7, axes7 = plt.subplots(1, 2, figsize=(12, 5))

# Box plot
sns.boxplot(x='satisfaction', y='Flight Distance', data=train_df, ax=axes7[0], hue='satisfaction', palette=['#ff7f0e', '#1f77b4'], legend=False)
axes7[0].set_xlabel('Satisfaction')
axes7[0].set_ylabel('Flight Distance (miles)')
axes7[0].set_title('Flight Distance by Satisfaction')
axes7[0].set_yscale('log')

# Bar chart by distance category
distance_counts_df = train_df.groupby(['distance_category', 'satisfaction']).size().unstack()
distance_counts_df.plot(kind='bar', stacked=True, ax=axes7[1], color=['#ff7f0e', '#1f77b4'])
axes7[1].set_xlabel('Distance Category')
axes7[1].set_ylabel('Count')
axes7[1].set_title('Satisfaction by Flight Distance Category')
axes7[1].legend(title='Satisfaction')
axes7[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/flight_distance_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: figures/flight_distance_analysis.png")
plt.close(fig7)

# ============================================================================
# 9. Statistical tests
# ============================================================================
print("\n9. Statistical tests...")

# 9.1 T-test for age difference
satisfied_age = train_df[train_df['satisfaction'] == 'satisfied']['Age']
neutral_age = train_df[train_df['satisfaction'] == 'neutral or dissatisfied']['Age']
t_stat_age, p_val_age = stats.ttest_ind(satisfied_age, neutral_age, equal_var=False)

print(f"\nAge difference between satisfied and neutral/dissatisfied:")
print(f"  Satisfied mean: {satisfied_age.mean():.2f} years")
print(f"  Neutral mean: {neutral_age.mean():.2f} years")
print(f"  T-statistic: {t_stat_age:.4f}")
print(f"  P-value: {p_val_age:.4f}")
print(f"  Significant difference: {'YES' if p_val_age < 0.05 else 'NO'}")

# 9.2 T-test for departure delay
satisfied_dep_delay = train_df[train_df['satisfaction'] == 'satisfied']['Departure Delay in Minutes']
neutral_dep_delay = train_df[train_df['satisfaction'] == 'neutral or dissatisfied']['Departure Delay in Minutes']
t_stat_delay, p_val_delay = stats.ttest_ind(satisfied_dep_delay, neutral_dep_delay, equal_var=False)

print(f"\nDeparture delay difference between satisfied and neutral/dissatisfied:")
print(f"  Satisfied mean: {satisfied_dep_delay.mean():.2f} minutes")
print(f"  Neutral mean: {neutral_dep_delay.mean():.2f} minutes")
print(f"  T-statistic: {t_stat_delay:.4f}")
print(f"  P-value: {p_val_delay:.4f}")
print(f"  Significant difference: {'YES' if p_val_delay < 0.05 else 'NO'}")

# ============================================================================
# 10. Summary insights
# ============================================================================
print("\n" + "="*80)
print("SUMMARY INSIGHTS FROM ADVANCED EDA")
print("="*80)

print("\n1. **Demographic Insights:**")
print(f"   - Highest satisfaction: {class_satisfaction.idxmax()} class ({class_satisfaction.max():.1f}%)")
print(f"   - Lowest satisfaction: {class_satisfaction.idxmin()} class ({class_satisfaction.min():.1f}%)")
print(f"   - Business travelers: {travel_satisfaction['Business travel']:.1f}% satisfied")
print(f"   - Personal travelers: {travel_satisfaction['Personal Travel']:.1f}% satisfied")

print("\n2. **Service Insights:**")
print(f"   - Largest rating gap: {rating_comparison.index[0]} ({rating_comparison['Difference'].iloc[0]:.2f} points)")
print(f"   - Smallest rating gap: {rating_comparison.index[-1]} ({rating_comparison['Difference'].iloc[-1]:.2f} points)")

print("\n3. **Operational Insights:**")
print(f"   - Average departure delay for satisfied: {satisfied_dep_delay.mean():.1f} minutes")
print(f"   - Average departure delay for neutral: {neutral_dep_delay.mean():.1f} minutes")
print(f"   - Delay difference is {'statistically significant' if p_val_delay < 0.05 else 'not statistically significant'}")

print("\n4. **Interaction Insights:**")
print(f"   - Best combination: {pivot_table.max().idxmax()} travel in {pivot_table.idxmax()[pivot_table.max().idxmax()]} class ({pivot_table.max().max():.1f}%)")
print(f"   - Worst combination: {pivot_table.min().idxmin()} travel in {pivot_table.idxmin()[pivot_table.min().idxmin()]} class ({pivot_table.min().min():.1f}%)")

print("\n5. **Correlation Insights:**")
print(f"   - Most positively correlated: {correlations.index[0]} ({correlations.iloc[0]:.3f})")
print(f"   - Most negatively correlated: {correlations.index[-1]} ({correlations.iloc[-1]:.3f})")

print("\n" + "="*80)
print("ADVANCED EDA COMPLETED")
print("="*80)
print(f"\nGenerated {8} new visualizations in figures/ directory")
print(f"Generated {2} new reports in reports/ directory")

# Clean up temporary columns
train_df = train_df.drop(columns=['satisfaction_encoded', 'age_group', 'distance_category'])

print("\nAll advanced EDA analyses completed successfully!")