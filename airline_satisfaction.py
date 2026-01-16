import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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