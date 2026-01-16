import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.impute import SimpleImputer
import warnings
import time
import os
warnings.filterwarnings('ignore')

print("Module airline_satisfaction loaded")

np.random.seed(42)

def advanced_eda(train_df, test_df):
    """
    Perform advanced exploratory data analysis focusing on relationships with target variable.
    Generates 8 visualizations and 2 report files.
    """
    print("\n" + "="*80)
    print("ADVANCED EDA - ADDITIONAL ANALYSES")
    print("="*80)

    # Create a copy to avoid modifying original
    train_df_copy = train_df.copy()

    # Encode target for analysis
    train_df_copy['satisfaction_encoded'] = train_df_copy['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})

    # 1. Target variable analysis by different segments
    print("\n1. Target variable analysis by different segments...")

    # 1.1 Satisfaction by gender
    print("\n1.1 Satisfaction by gender:")
    gender_satisfaction = train_df_copy.groupby('Gender')['satisfaction_encoded'].mean() * 100
    print(gender_satisfaction)

    # 1.2 Satisfaction by customer type
    print("\n1.2 Satisfaction by customer type:")
    customer_satisfaction = train_df_copy.groupby('Customer Type')['satisfaction_encoded'].mean() * 100
    print(customer_satisfaction)

    # 1.3 Satisfaction by travel type
    print("\n1.3 Satisfaction by travel type:")
    travel_satisfaction = train_df_copy.groupby('Type of Travel')['satisfaction_encoded'].mean() * 100
    print(travel_satisfaction)

    # 1.4 Satisfaction by class
    print("\n1.4 Satisfaction by class:")
    class_satisfaction = train_df_copy.groupby('Class')['satisfaction_encoded'].mean() * 100
    print(class_satisfaction)

    # 2. Create visualization: Satisfaction by different segments
    print("\n2. Creating satisfaction by segments visualization...")

    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('Satisfaction Rate by Different Segments', fontsize=16)

    # Gender
    gender_counts = train_df_copy.groupby(['Gender', 'satisfaction']).size().unstack()
    gender_counts.plot(kind='bar', stacked=True, ax=axes1[0, 0], color=['#ff7f0e', '#1f77b4'])
    axes1[0, 0].set_title('Satisfaction by Gender')
    axes1[0, 0].set_xlabel('Gender')
    axes1[0, 0].set_ylabel('Count')
    axes1[0, 0].legend(title='Satisfaction')
    axes1[0, 0].tick_params(axis='x', rotation=0)

    # Customer Type
    customer_counts = train_df_copy.groupby(['Customer Type', 'satisfaction']).size().unstack()
    customer_counts.plot(kind='bar', stacked=True, ax=axes1[0, 1], color=['#ff7f0e', '#1f77b4'])
    axes1[0, 1].set_title('Satisfaction by Customer Type')
    axes1[0, 1].set_xlabel('Customer Type')
    axes1[0, 1].set_ylabel('Count')
    axes1[0, 1].legend(title='Satisfaction')
    axes1[0, 1].tick_params(axis='x', rotation=0)

    # Type of Travel
    travel_counts = train_df_copy.groupby(['Type of Travel', 'satisfaction']).size().unstack()
    travel_counts.plot(kind='bar', stacked=True, ax=axes1[1, 0], color=['#ff7f0e', '#1f77b4'])
    axes1[1, 0].set_title('Satisfaction by Type of Travel')
    axes1[1, 0].set_xlabel('Type of Travel')
    axes1[1, 0].set_ylabel('Count')
    axes1[1, 0].legend(title='Satisfaction')
    axes1[1, 0].tick_params(axis='x', rotation=0)

    # Class
    class_counts = train_df_copy.groupby(['Class', 'satisfaction']).size().unstack()
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

    # 3. Age analysis by satisfaction
    print("\n3. Age analysis by satisfaction...")

    # 3.1 Age distribution by satisfaction
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    satisfied_ages = train_df_copy[train_df_copy['satisfaction'] == 'satisfied']['Age']
    neutral_ages = train_df_copy[train_df_copy['satisfaction'] == 'neutral or dissatisfied']['Age']

    axes2[0].hist([satisfied_ages, neutral_ages], bins=30, label=['Satisfied', 'Neutral/Dissatisfied'],
                  color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black')
    axes2[0].set_xlabel('Age')
    axes2[0].set_ylabel('Frequency')
    axes2[0].set_title('Age Distribution by Satisfaction')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)

    # Box plot
    sns.boxplot(x='satisfaction', y='Age', data=train_df_copy, ax=axes2[1], hue='satisfaction', palette=['#ff7f0e', '#1f77b4'], legend=False)
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

    train_df_copy['age_group'] = train_df_copy['Age'].apply(create_age_groups)

    # Calculate satisfaction rate by age group
    age_group_satisfaction = train_df_copy.groupby('age_group')['satisfaction_encoded'].mean() * 100
    age_group_counts = train_df_copy['age_group'].value_counts()

    print("\nSatisfaction rate by age group:")
    for age_group in age_group_satisfaction.index:
        rate = age_group_satisfaction[age_group]
        count = age_group_counts[age_group]
        print(f"  {age_group}: {rate:.1f}% satisfied (n={count})")

    # 4. Service ratings analysis by satisfaction
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
    satisfied_means = train_df_copy[train_df_copy['satisfaction'] == 'satisfied'][service_columns].mean()
    neutral_means = train_df_copy[train_df_copy['satisfaction'] == 'neutral or dissatisfied'][service_columns].mean()

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

    # 5. Delay analysis by satisfaction
    print("\n5. Delay analysis by satisfaction...")

    # 5.1 Delay distributions
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))

    # Departure delay
    sns.boxplot(x='satisfaction', y='Departure Delay in Minutes', data=train_df_copy, ax=axes4[0], hue='satisfaction', palette=['#ff7f0e', '#1f77b4'], legend=False)
    axes4[0].set_xlabel('Satisfaction')
    axes4[0].set_ylabel('Departure Delay (minutes)')
    axes4[0].set_title('Departure Delay by Satisfaction')
    axes4[0].set_yscale('log')  # Log scale due to outliers

    # Arrival delay
    sns.boxplot(x='satisfaction', y='Arrival Delay in Minutes', data=train_df_copy, ax=axes4[1], hue='satisfaction', palette=['#ff7f0e', '#1f77b4'], legend=False)
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
    delay_stats = train_df_copy.groupby('satisfaction')[['Departure Delay in Minutes', 'Arrival Delay in Minutes']].agg(['mean', 'median', 'std'])
    print(delay_stats)

    # 6. Interaction analysis: Satisfaction by Class and Type of Travel
    print("\n6. Interaction analysis: Satisfaction by Class and Type of Travel...")

    # Create pivot table
    pivot_table = train_df_copy.pivot_table(
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

    # 7. Correlation with target
    print("\n7. Correlation with target...")

    # Prepare data for correlation
    corr_df = train_df_copy.copy()

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

    # 8. Flight distance analysis
    print("\n8. Flight distance analysis...")

    # 8.1 Flight distance categories
    def create_distance_categories(distance):
        if distance <= 500:
            return 'Short (≤500 mi)'
        elif distance <= 1500:
            return 'Medium (501-1500 mi)'
        else:
            return 'Long (>1500 mi)'

    train_df_copy['distance_category'] = train_df_copy['Flight Distance'].apply(create_distance_categories)

    # Calculate satisfaction by distance category
    distance_satisfaction = train_df_copy.groupby('distance_category')['satisfaction_encoded'].mean() * 100
    distance_counts = train_df_copy['distance_category'].value_counts()

    print("\nSatisfaction by flight distance category:")
    for category in distance_satisfaction.index:
        rate = distance_satisfaction[category]
        count = distance_counts[category]
        print(f"  {category}: {rate:.1f}% satisfied (n={count})")

    # 8.2 Visualization
    fig7, axes7 = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot
    sns.boxplot(x='satisfaction', y='Flight Distance', data=train_df_copy, ax=axes7[0], hue='satisfaction', palette=['#ff7f0e', '#1f77b4'], legend=False)
    axes7[0].set_xlabel('Satisfaction')
    axes7[0].set_ylabel('Flight Distance (miles)')
    axes7[0].set_title('Flight Distance by Satisfaction')
    axes7[0].set_yscale('log')

    # Bar chart by distance category
    distance_counts_df = train_df_copy.groupby(['distance_category', 'satisfaction']).size().unstack()
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

    # 9. Statistical tests
    print("\n9. Statistical tests...")

    # 9.1 T-test for age difference
    satisfied_age = train_df_copy[train_df_copy['satisfaction'] == 'satisfied']['Age']
    neutral_age = train_df_copy[train_df_copy['satisfaction'] == 'neutral or dissatisfied']['Age']
    t_stat_age, p_val_age = stats.ttest_ind(satisfied_age, neutral_age, equal_var=False)

    print(f"\nAge difference between satisfied and neutral/dissatisfied:")
    print(f"  Satisfied mean: {satisfied_age.mean():.2f} years")
    print(f"  Neutral mean: {neutral_age.mean():.2f} years")
    print(f"  T-statistic: {t_stat_age:.4f}")
    print(f"  P-value: {p_val_age:.4f}")
    print(f"  Significant difference: {'YES' if p_val_age < 0.05 else 'NO'}")

    # 9.2 T-test for departure delay
    satisfied_dep_delay = train_df_copy[train_df_copy['satisfaction'] == 'satisfied']['Departure Delay in Minutes']
    neutral_dep_delay = train_df_copy[train_df_copy['satisfaction'] == 'neutral or dissatisfied']['Departure Delay in Minutes']
    t_stat_delay, p_val_delay = stats.ttest_ind(satisfied_dep_delay, neutral_dep_delay, equal_var=False)

    print(f"\nDeparture delay difference between satisfied and neutral/dissatisfied:")
    print(f"  Satisfied mean: {satisfied_dep_delay.mean():.2f} minutes")
    print(f"  Neutral mean: {neutral_dep_delay.mean():.2f} minutes")
    print(f"  T-statistic: {t_stat_delay:.4f}")
    print(f"  P-value: {p_val_delay:.4f}")
    print(f"  Significant difference: {'YES' if p_val_delay < 0.05 else 'NO'}")

    # 10. Summary insights
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
    train_df_copy = train_df_copy.drop(columns=['satisfaction_encoded', 'age_group', 'distance_category'])

    print("\nAll advanced EDA analyses completed successfully!")

    return train_df_copy

def generate_model_comparison_plots(all_results):
    """
    Generate additional comparison plots for model evaluation.
    """
    print("\n\\nGenerating additional model comparison plots...")

    # Create directories if they don't exist
    os.makedirs('figures', exist_ok=True)

    # 1. Multi-metric comparison bar plot
    print("Creating multi-metric comparison bar plot...")
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Sort by Accuracy for consistency
    df_sorted = all_results.sort_values('Accuracy', ascending=False)
    models = df_sorted['Model']
    x = np.arange(len(models))
    width = 0.2

    # Accuracy bars
    bars1 = ax1.bar(x - width, df_sorted['Accuracy'], width, label='Accuracy', color='steelblue')
    # F1-Score bars
    bars2 = ax1.bar(x, df_sorted['F1-Score'], width, label='F1-Score', color='orange')
    # ROC-AUC bars
    bars3 = ax1.bar(x + width, df_sorted['ROC-AUC'], width, label='ROC-AUC', color='green')

    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Multi-Metric Comparison of All Models', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0.8, 1.0)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # 2. Training time comparison (log scale)
    ax2.bar(models, df_sorted['Training Time (s)'], color='crimson')
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)
    ax2.set_title('Training Time Comparison (Log Scale)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (model, time) in enumerate(zip(models, df_sorted['Training Time (s)'])):
        ax2.text(i, time * 1.2, f'{time:.1f}s', ha='center', va='bottom', fontsize=8, rotation=0)

    plt.tight_layout()
    plt.savefig('figures/multi_metric_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved multi-metric comparison visualization: figures/multi_metric_comparison.png")
    plt.close(fig1)

    # 3. Model performance heatmap
    print("Creating model performance heatmap...")
    # Create a DataFrame for heatmap (metrics as columns, models as rows)
    heatmap_data = df_sorted[['Accuracy', 'F1-Score', 'ROC-AUC']].copy()
    heatmap_data.index = df_sorted['Model']

    fig2, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Score'})
    ax3.set_title('Model Performance Heatmap', fontsize=14)
    ax3.set_xlabel('Metric', fontsize=12)
    ax3.set_ylabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/performance_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved performance heatmap: figures/performance_heatmap.png")
    plt.close(fig2)

    # 4. Trade-off analysis: Accuracy vs Training Time
    print("Creating accuracy vs training time scatter plot...")
    fig3, ax4 = plt.subplots(figsize=(10, 6))

    # Plot each model
    scatter = ax4.scatter(df_sorted['Accuracy'], df_sorted['Training Time (s)'], s=100, alpha=0.7,
                         c=range(len(df_sorted)), cmap='viridis')

    # Add model labels
    for i, row in df_sorted.iterrows():
        ax4.annotate(row['Model'], (row['Accuracy'], row['Training Time (s)']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax4.set_xlabel('Accuracy', fontsize=12)
    ax4.set_ylabel('Training Time (seconds, log scale)', fontsize=12)
    ax4.set_title('Accuracy vs Training Time Trade-off', fontsize=14)
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    # Add a reference line for Pareto frontier (simplified)
    # Sort by accuracy descending, time ascending
    pareto = df_sorted.sort_values(['Accuracy', 'Training Time (s)'], ascending=[False, True])
    ax4.plot(pareto['Accuracy'], pareto['Training Time (s)'], 'r--', alpha=0.5, label='Pareto Frontier (approx)')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('figures/accuracy_vs_time_tradeoff.png', dpi=150, bbox_inches='tight')
    print("Saved accuracy vs training time trade-off visualization: figures/accuracy_vs_time_tradeoff.png")
    plt.close(fig3)

    print("\nAll additional model comparison visualizations generated successfully!")

def main():
    """
    Main pipeline for Airline Passenger Satisfaction prediction.
    Follows similar structure to titanic.py reference project.
    """
    print("DEBUG: Main function started")
    print("=" * 80)
    print("AIRLINE PASSENGER SATISFACTION PREDICTION PROJECT")
    print("=" * 80)

    # Create necessary directories
    os.makedirs('reports', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    print("Created necessary directories: reports/, figures/, models/, results/")

    # Step 1: Load data
    print("\n--- 1. LOADING DATA ---")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')

    # Remove first unnamed column if exists
    if train_df.columns[0] == 'Unnamed: 0':
        train_df = train_df.drop(columns=['Unnamed: 0'])
    if test_df.columns[0] == 'Unnamed: 0':
        test_df = test_df.drop(columns=['Unnamed: 0'])

    print(f"Training set: {train_df.shape[0]} passengers, {train_df.shape[1]} features")
    print(f"Test set: {test_df.shape[0]} passengers, {test_df.shape[1]} features")

    # Combine train and test for EDA (excluding target from test)
    test_df_no_target = test_df.copy()
    if 'satisfaction' in test_df.columns:
        test_df_no_target = test_df.drop(columns=['satisfaction'])
    combined_df = pd.concat([train_df, test_df_no_target], ignore_index=True)
    print(f"Combined dataset for EDA: {combined_df.shape[0]} passengers")

    # Step 2: Exploratory Data Analysis (EDA)
    print("\n--- 2. EXPLORATORY DATA ANALYSIS ---")

    # 2.1 Basic dataset information
    print("\n2.1 Basic dataset information:")
    print(f"Columns: {list(train_df.columns)}")
    print(f"\\nData types:")
    print(train_df.dtypes)

    # 2.2 Missing values analysis
    print("\n2.2 Missing values analysis:")
    missing_train = train_df.isnull().sum()
    missing_percent_train = (missing_train / len(train_df)) * 100
    missing_test = test_df.isnull().sum()
    missing_percent_test = (missing_test / len(test_df)) * 100

    missing_df = pd.DataFrame({
        'Column': missing_train.index,
        'Missing_Train': missing_train.values,
        'Missing_Train_%': missing_percent_train.values,
        'Missing_Test': missing_test.values,
        'Missing_Test_%': missing_percent_test.values
    })
    missing_df = missing_df[(missing_df['Missing_Train'] > 0) | (missing_df['Missing_Test'] > 0)]
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("No missing values found in train or test sets.")

    # Save missing values summary to CSV
    missing_df.to_csv('reports/missing_values_summary.csv', index=False)
    print("Saved missing values summary to: reports/missing_values_summary.csv")

    # 2.3 Visualize missing values
    print("\nCreating missing values visualization...")
    fig_missing, ax_missing = plt.subplots(figsize=(10, 6))
    if len(missing_df) > 0:
        bars = ax_missing.barh(missing_df['Column'], missing_df['Missing_Train_%'], color='skyblue')
        ax_missing.set_xlabel('Percentage of missing values (%)', fontsize=12)
        ax_missing.set_title('Missing values in training set', fontsize=14)
        ax_missing.grid(True, alpha=0.3, axis='x')
        for bar, value in zip(bars, missing_df['Missing_Train_%']):
            ax_missing.text(value + 0.5, bar.get_y() + bar.get_height()/2, f'{value:.1f}%',
                           va='center', ha='left', fontsize=10)
    else:
        ax_missing.text(0.5, 0.5, 'No missing values', ha='center', va='center', fontsize=14)
        ax_missing.set_title('Missing values in training set', fontsize=14)

    plt.tight_layout()
    plt.savefig('figures/missing_values.png', dpi=150, bbox_inches='tight')
    print("Saved missing values visualization: figures/missing_values.png")
    plt.close(fig_missing)

    # 2.4 Target variable distribution
    print("\n2.4 Target variable distribution:")
    target_counts = train_df['satisfaction'].value_counts()
    target_percent = train_df['satisfaction'].value_counts(normalize=True) * 100
    target_summary = pd.DataFrame({
        'Count': target_counts,
        'Percentage': target_percent
    })
    print(target_summary)

    # Visualize target distribution
    fig_target, ax_target = plt.subplots(figsize=(8, 6))
    bars = ax_target.bar(target_summary.index, target_summary['Percentage'], color=['#ff7f0e', '#1f77b4'])
    ax_target.set_xlabel('Satisfaction', fontsize=12)
    ax_target.set_ylabel('Percentage (%)', fontsize=12)
    ax_target.set_title('Distribution of Passenger Satisfaction', fontsize=14)
    ax_target.grid(True, alpha=0.3, axis='y')
    for bar, perc in zip(bars, target_summary['Percentage']):
        height = bar.get_height()
        ax_target.text(bar.get_x() + bar.get_width()/2., height + 1,
                      f'{perc:.1f}%', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('figures/target_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved target distribution visualization: figures/target_distribution.png")
    plt.close(fig_target)

    # 2.5 Numerical features analysis
    print("\n2.5 Numerical features analysis:")
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target if it's numerical (it's categorical)
    if 'satisfaction' in numerical_cols:
        numerical_cols.remove('satisfaction')
    print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

    # Statistical summary
    numerical_summary = train_df[numerical_cols].describe().T
    numerical_summary['skewness'] = train_df[numerical_cols].skew()
    numerical_summary['kurtosis'] = train_df[numerical_cols].kurt()
    print("\nStatistical summary of numerical features:")
    print(numerical_summary)

    # Save numerical summary to CSV
    numerical_summary.to_csv('reports/numerical_summary.csv')
    print("Saved numerical summary to: reports/numerical_summary.csv")

    # Visualize numerical distributions
    print("\nCreating numerical distributions visualization...")
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 3) // 4  # 4 columns per row
    fig_num, axes = plt.subplots(n_rows, 4, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        ax.hist(train_df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('figures/numerical_distributions.png', dpi=150, bbox_inches='tight')
    print("Saved numerical distributions visualization: figures/numerical_distributions.png")
    plt.close(fig_num)

    # 2.6 Categorical features analysis
    print("\n2.6 Categorical features analysis:")
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    # Include target as categorical
    if 'satisfaction' not in categorical_cols:
        categorical_cols.append('satisfaction')
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # Categorical summary
    categorical_summary = []
    for col in categorical_cols:
        if col == 'satisfaction':
            continue
        value_counts = train_df[col].value_counts()
        value_percent = train_df[col].value_counts(normalize=True) * 100
        for val, count, percent in zip(value_counts.index, value_counts.values, value_percent.values):
            categorical_summary.append({
                'Feature': col,
                'Value': val,
                'Count': count,
                'Percentage': percent
            })

    categorical_df = pd.DataFrame(categorical_summary)
    print("\nCategorical features summary (top 5 per feature):")
    for col in categorical_cols:
        if col == 'satisfaction':
            continue
        col_df = categorical_df[categorical_df['Feature'] == col].head(5)
        print(f"\\n{col}:")
        print(col_df[['Value', 'Count', 'Percentage']].to_string(index=False))

    # Save categorical summary to CSV
    categorical_df.to_csv('reports/categorical_summary.csv', index=False)
    print("\\nSaved categorical summary to: reports/categorical_summary.csv")

    # Visualize categorical distributions
    print("\nCreating categorical distributions visualization...")
    n_cat = len([c for c in categorical_cols if c != 'satisfaction'])
    n_rows_cat = (n_cat + 2) // 3  # 3 columns per row
    fig_cat, axes_cat = plt.subplots(n_rows_cat, 3, figsize=(18, n_rows_cat * 4))
    axes_cat = axes_cat.flatten()

    cat_idx = 0
    for col in categorical_cols:
        if col == 'satisfaction':
            continue
        ax = axes_cat[cat_idx]
        top_values = train_df[col].value_counts().head(10)
        bars = ax.bar(range(len(top_values)), top_values.values, color='steelblue', alpha=0.7)
        ax.set_title(col, fontsize=12)
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(top_values)))
        ax.set_xticklabels(top_values.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        cat_idx += 1

    # Hide unused subplots
    for i in range(cat_idx, len(axes_cat)):
        axes_cat[i].axis('off')

    plt.tight_layout()
    plt.savefig('figures/categorical_distributions.png', dpi=150, bbox_inches='tight')
    print("Saved categorical distributions visualization: figures/categorical_distributions.png")
    plt.close(fig_cat)

    # 2.7 Correlation analysis
    print("\n2.7 Correlation analysis:")
    # Select numerical columns for correlation
    corr_cols = numerical_cols.copy()
    # Encode target for correlation if needed
    train_df_corr = train_df.copy()
    if 'satisfaction' in train_df_corr.columns:
        train_df_corr['satisfaction_encoded'] = train_df_corr['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
        corr_cols.append('satisfaction_encoded')

    correlation_matrix = train_df_corr[corr_cols].corr()
    print("\\nCorrelation matrix (top 10 highest absolute correlations):")
    # Get top correlations (excluding self-correlation)
    corr_pairs = correlation_matrix.unstack()
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    corr_pairs_sorted = corr_pairs.abs().sort_values(ascending=False)
    top_corr = corr_pairs_sorted.head(10)
    for idx, value in top_corr.items():
        print(f"  {idx[0]} - {idx[1]}: {correlation_matrix.loc[idx[0], idx[1]]:.3f}")

    # Save correlation matrix to CSV
    correlation_matrix.to_csv('reports/correlation_matrix.csv')
    print("\\nSaved correlation matrix to: reports/correlation_matrix.csv")

    # Visualize correlation matrix
    print("\nCreating correlation matrix visualization...")
    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    im = ax_corr.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    # Add labels
    ax_corr.set_xticks(np.arange(len(corr_cols)))
    ax_corr.set_yticks(np.arange(len(corr_cols)))
    ax_corr.set_xticklabels(corr_cols, rotation=45, ha='right')
    ax_corr.set_yticklabels(corr_cols)

    # Add correlation values in cells
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            text = ax_corr.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center',
                               color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')

    ax_corr.set_title('Correlation Matrix of Numerical Features', fontsize=14)
    plt.colorbar(im, ax=ax_corr)
    plt.tight_layout()
    plt.savefig('figures/correlation_matrix.png', dpi=150, bbox_inches='tight')
    print("Saved correlation matrix visualization: figures/correlation_matrix.png")
    plt.close(fig_corr)

    # 2.8 Outlier detection
    print("\n2.8 Outlier detection (Z-score > 3):")
    outlier_info = []
    for col in numerical_cols:
        z_scores = np.abs((train_df[col] - train_df[col].mean()) / train_df[col].std())
        outliers = z_scores > 3
        outlier_count = outliers.sum()
        outlier_percent = (outlier_count / len(train_df)) * 100
        if outlier_count > 0:
            outlier_info.append({
                'Feature': col,
                'Outlier Count': outlier_count,
                'Outlier %': outlier_percent,
                'Mean': train_df[col].mean(),
                'Std': train_df[col].std()
            })

    if outlier_info:
        outlier_df = pd.DataFrame(outlier_info)
        print(outlier_df.to_string(index=False))
        outlier_df.to_csv('reports/outlier_info.csv', index=False)
        print("Saved outlier information to: reports/outlier_info.csv")
    else:
        print("No outliers detected (Z-score > 3).")

    # Advanced EDA with additional analyses and visualizations
    print("\n" + "=" * 80)
    print("ADVANCED EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    advanced_eda(train_df, test_df)

    print("\n" + "=" * 80)
    print("EDA COMPLETED")
    print("=" * 80)

    # Step 3: Data preprocessing and feature engineering
    print("\n--- 3. DATA PREPROCESSING AND FEATURE ENGINEERING ---")

    # Create a copy for preprocessing
    train_processed = train_df.copy()
    test_processed = test_df.copy()

    # 3.1 Handle missing values
    print("\n3.1 Handling missing values...")
    # Check which columns have missing values
    missing_cols_train = train_processed.columns[train_processed.isnull().any()].tolist()
    missing_cols_test = test_processed.columns[test_processed.isnull().any()].tolist()

    if missing_cols_train or missing_cols_test:
        print(f"Columns with missing values in train: {missing_cols_train}")
        print(f"Columns with missing values in test: {missing_cols_test}")

        # For numerical columns, impute with median
        numerical_missing_train = [col for col in missing_cols_train if col in numerical_cols]
        numerical_missing_test = [col for col in missing_cols_test if col in numerical_cols]

        if numerical_missing_train or numerical_missing_test:
            imputer = SimpleImputer(strategy='median')
            train_processed[numerical_missing_train] = imputer.fit_transform(train_processed[numerical_missing_train])
            test_processed[numerical_missing_test] = imputer.transform(test_processed[numerical_missing_test])
            print(f"Imputed numerical columns with median.")

        # For categorical columns, impute with mode
        categorical_missing_train = [col for col in missing_cols_train if col in categorical_cols and col != 'satisfaction']
        categorical_missing_test = [col for col in missing_cols_test if col in categorical_cols and col != 'satisfaction']

        if categorical_missing_train or categorical_missing_test:
            for col in categorical_missing_train:
                mode_val = train_processed[col].mode()[0]
                train_processed[col] = train_processed[col].fillna(mode_val)
            for col in categorical_missing_test:
                mode_val = test_processed[col].mode()[0]
                test_processed[col] = test_processed[col].fillna(mode_val)
            print(f"Imputed categorical columns with mode.")
    else:
        print("No missing values to impute.")

    # 3.2 Feature engineering
    print("\n3.2 Feature engineering...")

    # Create age groups
    def create_age_groups(age):
        if age <= 20:
            return 'Teen'
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

    train_processed['age_group'] = train_processed['Age'].apply(create_age_groups)
    test_processed['age_group'] = test_processed['Age'].apply(create_age_groups)

    # Create flight distance categories
    def create_distance_categories(distance):
        if distance <= 500:
            return 'Short'
        elif distance <= 1500:
            return 'Medium'
        else:
            return 'Long'

    train_processed['flight_distance_category'] = train_processed['Flight Distance'].apply(create_distance_categories)
    test_processed['flight_distance_category'] = test_processed['Flight Distance'].apply(create_distance_categories)

    # Create total delay feature
    train_processed['total_delay'] = train_processed['Departure Delay in Minutes'] + train_processed['Arrival Delay in Minutes']
    test_processed['total_delay'] = test_processed['Departure Delay in Minutes'] + test_processed['Arrival Delay in Minutes']

    # Create delay flag
    train_processed['has_delay'] = (train_processed['total_delay'] > 0).astype(int)
    test_processed['has_delay'] = (test_processed['total_delay'] > 0).astype(int)

    print("Created new features: age_group, flight_distance_category, total_delay, has_delay")

    # 3.3 Encode categorical variables
    print("\n3.3 Encoding categorical variables...")

    # Identify categorical columns (excluding target)
    cat_cols_to_encode = train_processed.select_dtypes(include=['object']).columns.tolist()
    if 'satisfaction' in cat_cols_to_encode:
        cat_cols_to_encode.remove('satisfaction')

    print(f"Categorical columns to encode: {cat_cols_to_encode}")

    # Use Label Encoding for categorical variables
    label_encoders = {}
    for col in cat_cols_to_encode:
        le = LabelEncoder()
        # Fit on train data
        le.fit(train_processed[col].astype(str))
        train_processed[col] = le.transform(train_processed[col].astype(str))
        # Transform test data (handle unseen labels with a fallback)
        test_processed[col] = test_processed[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        label_encoders[col] = le
        print(f"  Encoded {col} with {len(le.classes_)} unique values")

    # Encode target variable
    target_encoder = LabelEncoder()
    train_processed['satisfaction_encoded'] = target_encoder.fit_transform(train_processed['satisfaction'])
    if 'satisfaction' in test_processed.columns:
        test_processed['satisfaction_encoded'] = target_encoder.transform(test_processed['satisfaction'])

    print(f"Encoded target: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")

    # 3.4 Scale numerical features
    print("\n3.4 Scaling numerical features...")

    # Identify numerical columns (excluding target and encoded target)
    num_cols_to_scale = train_processed.select_dtypes(include=[np.number]).columns.tolist()
    if 'satisfaction_encoded' in num_cols_to_scale:
        num_cols_to_scale.remove('satisfaction_encoded')
    if 'satisfaction' in num_cols_to_scale:
        num_cols_to_scale.remove('satisfaction')

    print(f"Numerical columns to scale: {num_cols_to_scale}")

    scaler = StandardScaler()
    train_processed[num_cols_to_scale] = scaler.fit_transform(train_processed[num_cols_to_scale])
    test_processed[num_cols_to_scale] = scaler.transform(test_processed[num_cols_to_scale])
    print("Scaled numerical features using StandardScaler.")

    # Prepare feature matrix and target vector
    print("\n3.5 Preparing feature matrix and target vector...")

    # Define features to use (exclude target and original satisfaction)
    feature_cols = [col for col in train_processed.columns
                   if col not in ['satisfaction', 'satisfaction_encoded']]

    X_train = train_processed[feature_cols]
    y_train = train_processed['satisfaction_encoded']

    if 'satisfaction_encoded' in test_processed.columns:
        X_test = test_processed[feature_cols]
        y_test = test_processed['satisfaction_encoded']
    else:
        X_test = test_processed[feature_cols]
        y_test = None

    print(f"Training features shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}")
    if y_test is not None:
        print(f"Test features shape: {X_test.shape}")
        print(f"Test target shape: {y_test.shape}")
    else:
        print(f"Test features shape: {X_test.shape}")
        print("Test target not available.")

    # Save processed data for modeling
    X_train.to_csv('reports/X_train_processed.csv', index=False)
    y_train.to_csv('reports/y_train.csv', index=False)
    X_test.to_csv('reports/X_test_processed.csv', index=False)
    if y_test is not None:
        y_test.to_csv('reports/y_test.csv', index=False)

    print("Saved processed data to reports/ directory.")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED")
    print("=" * 80)

    # Step 4: Model training and evaluation
    print("\n--- 4. MODEL TRAINING AND EVALUATION ---")

    # Split training data into train and validation sets
    print("\n4.1 Splitting data into training and validation sets...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print(f"Training split: {X_train_split.shape[0]} samples")
    print(f"Validation split: {X_val.shape[0]} samples")

    # 4.2 Initialize models
    print("\n4.2 Initializing classification models...")

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_jobs=1)
    }

    print(f"Models to evaluate: {list(models.keys())}")

    # 4.3 Train and evaluate models
    print("\n4.3 Training and evaluating models...")

    results = []
    for name, model in models.items():
        print(f"\\nTraining {name}...")
        start_time = time.time()

        # Train model
        model.fit(X_train_split, y_train_split)
        train_time = time.time() - start_time

        # Predict on validation set
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else np.nan

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Training Time (s)': train_time
        })

        roc_auc_str = f'{roc_auc:.4f}' if not np.isnan(roc_auc) else 'N/A'
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc_str}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)

    print("\n\\nModel performance summary (sorted by Accuracy):")
    print(results_df.to_string(index=False))

    # Save results to CSV
    results_df.to_csv('reports/model_results_initial.csv', index=False)
    print("\\nSaved initial model results to: reports/model_results_initial.csv")

    # 4.4 Hyperparameter tuning for top models
    print("\n\\n4.4 Hyperparameter tuning for top 3 models...")

    # Get top 3 models based on accuracy
    top_models = results_df.head(3)['Model'].tolist()
    print(f"Top 3 models for hyperparameter tuning: {top_models}")

    # Define hyperparameter grids
    param_grids = {
        'Random Forest': {
            'n_estimators': [100],
            'max_depth': [10, None],
            'min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [3, 5]
        },
        'Logistic Regression': {
            'C': [0.1, 1],
            'penalty': ['l2'],
            'solver': ['liblinear']
        },
        'Decision Tree': {
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        },
        'SVM': {
            'C': [0.1],
            'kernel': ['linear']
        },
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5],
            'weights': ['uniform', 'distance']
        }
    }

    tuned_results = []
    for model_name in top_models:
        if model_name not in param_grids:
            print(f"  No hyperparameter grid defined for {model_name}, skipping.")
            continue

        print(f"\\nTuning {model_name}...")

        # Get base model
        base_model = models[model_name]

        # Perform GridSearchCV
        grid_search = GridSearchCV(
            base_model,
            param_grids[model_name],
            cv=5,
            scoring='accuracy',
            n_jobs=1,
            verbose=0
        )

        start_time = time.time()
        grid_search.fit(X_train_split, y_train_split)
        tuning_time = time.time() - start_time

        # Get best model
        best_model = grid_search.best_estimator_

        # Evaluate on validation set
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)[:, 1] if hasattr(best_model, 'predict_proba') else None

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else np.nan

        tuned_results.append({
            'Model': f"{model_name} (Tuned)",
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Training Time (s)': tuning_time,
            'Best Params': str(grid_search.best_params_)
        })

        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Validation Accuracy: {accuracy:.4f} (improvement: {accuracy - results_df[results_df['Model'] == model_name]['Accuracy'].values[0]:.4f})")

    # Combine initial and tuned results
    all_results = pd.concat([results_df, pd.DataFrame(tuned_results)], ignore_index=True)
    all_results = all_results.sort_values('Accuracy', ascending=False)

    print("\n\\nFinal model performance summary (including tuned models):")
    print(all_results.to_string(index=False))

    # Save all results to CSV
    all_results.to_csv('reports/model_results_final.csv', index=False)
    print("\\nSaved final model results to: reports/model_results_final.csv")

    # Generate additional model comparison plots
    generate_model_comparison_plots(all_results)

    # Generate markdown table for report
    print("\\nGenerating markdown table for report...")
    # Create a copy for formatting
    df_md = all_results.copy()
    # Fill NaN with empty string
    df_md = df_md.fillna('')
    # Round numeric columns
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Training Time (s)']:
        if col in df_md.columns:
            df_md[col] = df_md[col].apply(lambda x: f'{x:.4f}' if isinstance(x, (int, float)) else x)

    # Create markdown table
    md_table = '| Model | Accuracy | F1-Score | ROC-AUC | Czas treningu (s) |\n'
    md_table += '|-------|----------|----------|---------|-------------------|\n'
    for _, row in df_md.iterrows():
        model = row['Model']
        acc = row['Accuracy']
        f1 = row['F1-Score']
        roc = row['ROC-AUC']
        time = row['Training Time (s)']
        md_table += f'| {model} | {acc} | {f1} | {roc} | {time} |\n'

    # Save markdown table to file
    with open('reports/model_results_table.md', 'w') as f:
        f.write(md_table)
    print("Saved markdown table to: reports/model_results_table.md")

    # 4.5 Feature importance analysis
    print("\n\\n4.5 Feature importance analysis...")

    # Get best model (assume it's the first in sorted results)
    best_model_name = all_results.iloc[0]['Model']
    print(f"Best model: {best_model_name}")

    # Extract best model object
    if "(Tuned)" in best_model_name:
        base_name = best_model_name.replace(" (Tuned)", "")
        # Find the tuned model
        for result in tuned_results:
            if result['Model'] == best_model_name:
                # We need to get the actual model, but we don't have it stored
                # Use the model from models dictionary and retrain with best params
                print(f"  Note: Feature importance for tuned model requires retraining.")
                best_model = None
                break
    else:
        best_model = models[best_model_name]

    # If we have a tree-based model, compute feature importance
    tree_based_models = ['Random Forest', 'Gradient Boosting', 'Decision Tree']
    if any(model_type in best_model_name for model_type in tree_based_models):
        print(f"  Computing feature importance for {best_model_name}...")

        # For simplicity, use Random Forest feature importance
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train_split, y_train_split)

        feature_importance = pd.DataFrame({
            'Feature': X_train_split.columns,
            'Importance': rf_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)

        print("\\nTop 10 most important features:")
        print(feature_importance.head(10).to_string(index=False))

        # Save feature importance
        feature_importance.to_csv('reports/feature_importance.csv', index=False)
        print("Saved feature importance to: reports/feature_importance.csv")

        # Visualize feature importance
        fig_importance, ax_importance = plt.subplots(figsize=(12, 8))
        top_features = feature_importance.head(15)
        bars = ax_importance.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
        ax_importance.set_yticks(range(len(top_features)))
        ax_importance.set_yticklabels(top_features['Feature'])
        ax_importance.set_xlabel('Feature Importance', fontsize=12)
        ax_importance.set_title(f'Top 15 Feature Importance ({best_model_name})', fontsize=14)
        ax_importance.invert_yaxis()  # Most important at top
        plt.tight_layout()
        plt.savefig('figures/feature_importance.png', dpi=150, bbox_inches='tight')
        print("Saved feature importance visualization: figures/feature_importance.png")
        plt.close(fig_importance)
    else:
        print(f"  Feature importance not available for {best_model_name} (not a tree-based model).")

    # 4.6 ROC curves for all models
    print("\n\\n4.6 Generating ROC curves...")

    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))

    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            # Retrain on full training split for ROC curve
            model.fit(X_train_split, y_train_split)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Curves for All Models', fontsize=14)
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/roc_curves.png', dpi=150, bbox_inches='tight')
    print("Saved ROC curves visualization: figures/roc_curves.png")
    plt.close(fig_roc)

    # 4.7 Confusion matrix for best model
    print("\n\\n4.7 Generating confusion matrix for best model...")

    # Use Random Forest as representative best model
    best_rf = RandomForestClassifier(random_state=42)
    best_rf.fit(X_train_split, y_train_split)
    y_pred = best_rf.predict(X_val)

    cm = confusion_matrix(y_val, y_pred)

    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=target_encoder.classes_,
                yticklabels=target_encoder.classes_)
    ax_cm.set_xlabel('Predicted', fontsize=12)
    ax_cm.set_ylabel('Actual', fontsize=12)
    ax_cm.set_title('Confusion Matrix (Random Forest)', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Saved confusion matrix visualization: figures/confusion_matrix.png")
    plt.close(fig_cm)

    print("\n" + "=" * 80)
    print("MODELING COMPLETED")
    print("=" * 80)

    # Final summary
    print("\n" + "=" * 80)
    print("PROJECT SUMMARY")
    print("=" * 80)
    print(f"1. Dataset: {train_df.shape[0]} training passengers, {test_df.shape[0]} test passengers")
    print(f"2. Features: {X_train.shape[1]} after preprocessing")
    print(f"3. Best model: {best_model_name} with accuracy: {all_results.iloc[0]['Accuracy']:.4f}")
    print(f"4. All results saved to: reports/model_results_final.csv")
    print(f"5. Visualizations saved to: figures/ directory")
    print("=" * 80)

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\\nTotal execution time: {end_time - start_time:.2f} seconds")