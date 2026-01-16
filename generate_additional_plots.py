import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories if they don't exist
os.makedirs('figures', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Data from the report table (manually entered)
# Model, Accuracy, F1-Score, ROC-AUC, Training Time (s)
model_data = [
    {'Model': 'Random Forest (Tuned)', 'Accuracy': 0.9642, 'F1-Score': 0.9641, 'ROC-AUC': 0.9944, 'Training Time (s)': 2135.0887},
    {'Model': 'Random Forest', 'Accuracy': 0.9641, 'F1-Score': 0.9640, 'ROC-AUC': 0.9943, 'Training Time (s)': 20.0139},
    {'Model': 'Extra Trees (Tuned)', 'Accuracy': 0.9628, 'F1-Score': 0.9627, 'ROC-AUC': 0.9940, 'Training Time (s)': 2854.4734},
    {'Model': 'Extra Trees', 'Accuracy': 0.9623, 'F1-Score': 0.9622, 'ROC-AUC': 0.9937, 'Training Time (s)': 13.5433},
    {'Model': 'SVM', 'Accuracy': 0.9538, 'F1-Score': 0.9537, 'ROC-AUC': 0.9891, 'Training Time (s)': 842.3743},
    {'Model': 'Decision Tree', 'Accuracy': 0.9462, 'F1-Score': 0.9462, 'ROC-AUC': 0.9458, 'Training Time (s)': 1.2279},
    {'Model': 'Gradient Boosting', 'Accuracy': 0.9454, 'F1-Score': 0.9453, 'ROC-AUC': 0.9886, 'Training Time (s)': 33.3738},
    {'Model': 'K-Nearest Neighbors', 'Accuracy': 0.9222, 'F1-Score': 0.9218, 'ROC-AUC': 0.9651, 'Training Time (s)': 0.0500},
    {'Model': 'AdaBoost', 'Accuracy': 0.9095, 'F1-Score': 0.9093, 'ROC-AUC': 0.9712, 'Training Time (s)': 6.3514},
    {'Model': 'Logistic Regression', 'Accuracy': 0.8775, 'F1-Score': 0.8773, 'ROC-AUC': 0.9294, 'Training Time (s)': 0.2406},
    {'Model': 'LDA', 'Accuracy': 0.8762, 'F1-Score': 0.8760, 'ROC-AUC': 0.9274, 'Training Time (s)': 0.3105},
    {'Model': 'Gaussian Naive Bayes', 'Accuracy': 0.8630, 'F1-Score': 0.8627, 'ROC-AUC': 0.9209, 'Training Time (s)': 0.0481},
    {'Model': 'QDA', 'Accuracy': 0.8560, 'F1-Score': 0.8556, 'ROC-AUC': 0.9182, 'Training Time (s)': 0.1480}
]

df = pd.DataFrame(model_data)

# Save the data to CSV for reference
df.to_csv('reports/model_results_extended.csv', index=False)
print("Saved extended model results to: reports/model_results_extended.csv")

# 1. Multi-metric comparison bar plot
print("Creating multi-metric comparison bar plot...")
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Sort by Accuracy for consistency
df_sorted = df.sort_values('Accuracy', ascending=False)
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
scatter = ax4.scatter(df['Accuracy'], df['Training Time (s)'], s=100, alpha=0.7, c=range(len(df)), cmap='viridis')

# Add model labels
for i, row in df.iterrows():
    ax4.annotate(row['Model'], (row['Accuracy'], row['Training Time (s)']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax4.set_xlabel('Accuracy', fontsize=12)
ax4.set_ylabel('Training Time (seconds, log scale)', fontsize=12)
ax4.set_title('Accuracy vs Training Time Trade-off', fontsize=14)
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)

# Add a reference line for Pareto frontier (simplified)
# Sort by accuracy descending, time ascending
pareto = df.sort_values(['Accuracy', 'Training Time (s)'], ascending=[False, True])
ax4.plot(pareto['Accuracy'], pareto['Training Time (s)'], 'r--', alpha=0.5, label='Pareto Frontier (approx)')
ax4.legend()

plt.tight_layout()
plt.savefig('figures/accuracy_vs_time_tradeoff.png', dpi=150, bbox_inches='tight')
print("Saved accuracy vs training time trade-off visualization: figures/accuracy_vs_time_tradeoff.png")
plt.close(fig3)

print("\nAll additional visualizations generated successfully!")
print("New figures created:")
print("  - figures/multi_metric_comparison.png")
print("  - figures/performance_heatmap.png")
print("  - figures/accuracy_vs_time_tradeoff.png")