import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from training_functions.balancing_data import balance_dataset
from training_functions.classifiers import Model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, recall_score, precision_score, \
    accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek, SMOTEENN
from training_functions.visuals import  plot_confusion_matrix
from collections import Counter
pd.set_option('display.max_columns', None)
import pandas as pd
import numpy as np
from training_functions.funnel_dictionary import get_sorted_funnel_for_


def read_and_prepare_for_training(starting_level):

    _df = pd.read_csv('training_data/preprocessed_df.tsv', sep="\t", index_col='id')
    _df['createdate'] = pd.to_datetime(_df['createdate'])
    _df['requested_bg_date'] = pd.to_datetime(_df['requested_bg_date'])
    _df['mql_date'] = pd.to_datetime(_df['mql_date'])
    _df['sql_date'] = pd.to_datetime(_df['sql_date'])
    _df['bg_enrolled_date'] = pd.to_datetime(_df['bg_enrolled_date'])
    
    if starting_level == "requested":
        _df = _df[_df['requested_bg'] == 'requested'].copy()
    if starting_level == "sql":
        _df = _df[_df['got_sql'] == 'sql'].copy()
    
    return _df
    
def split_to_current_and_future(_df):
    text_cond = (_df['why_do_you_want_to_start_a_career_in_tech'].fillna("") != "")
    highest_level_of_education_is_known = (_df['highest_level_of_education'].notna())
    in_contact_with_job_advisor_is_known = (_df['in_contact_with_job_advisor'].notna())
    preferred_cohort_is_known = (_df['preferred_cohort'].notna())
    lpvariant_is_known = (_df['lpvariant'].notna())
    
    conditions = text_cond & highest_level_of_education_is_known & in_contact_with_job_advisor_is_known & preferred_cohort_is_known
    conditions = text_cond & highest_level_of_education_is_known & in_contact_with_job_advisor_is_known
    conditions = text_cond & in_contact_with_job_advisor_is_known
    conditions = True
    
    _future_test = _df[(_df['mql_date'] >= cutoff) & conditions].copy()
    _current_df = _df[(_df['mql_date'] < cutoff) & (_df['mql_date'] >= start_date) & conditions].copy()
    return _current_df, _future_test

if __name__ == '__main__':
    cutoff = '2024-11-01'
    start_date = "2023-01-01"
    starting_level = "mql"
    target = 'enrolled'
    
    df = read_and_prepare_for_training(starting_level)
    
    features = [col for col in df.columns if col.startswith("feat")]
    text_features = [col for col in df.columns if col.startswith("svd")]
    all_features = text_features + features
    non_feature_col = [col for col in df.columns if col not in all_features]
    cat_cols, numerical = get_sorted_funnel_for_(starting_level, text_features)
    final_columns = numerical + cat_cols + [target]
    
    df = df[(df[non_feature_col].notna().sum(axis=1) / len(df.columns)) > 0.1].copy()
    
    # 2. Create a boolean column indicating whether row is "requested"
    df['is_requested'] = df['requested_bg'].eq('requested')
    df['enrolled'] = np.where(df['closed_won_deal__program_duration'].notna(), "enrolled", "didn't enroll")
    
    for col in cat_cols:
        df[col] = df[col].astype(str)
    
    current_df, future_test = split_to_current_and_future(df)
    
    pre_model = current_df.copy()
    pre_model = pre_model[final_columns].copy()
    pre_model_future = future_test[final_columns].copy()
    
    print(len(pre_model), list(pre_model[target].value_counts().items()))
    print(len(pre_model_future), list(pre_model_future[target].value_counts().items()))
    
    # Check the class distribution
    print("Original class distribution:")
    print(pre_model[target].value_counts())
    print(f"Imbalance ratio: 1:{pre_model[target].value_counts()[0] / pre_model[target].value_counts()[1]:.2f}")
    
    # Remove target from feature lists if present
    if target in cat_cols:
        cat_cols.remove(target)
    if target in numerical:
        numerical.remove(target)
    
    print(f"\nCategorical columns: {len(cat_cols)}")
    print(f"Numerical columns: {len(numerical)}")
    
    # First, determine if we need to encode the target
    if pre_model[target].dtype in ('object', "category", "string"):
        # Encode string labels to numbers
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(pre_model[target])
        # Check the mapping
        for i, label in enumerate(label_encoder.classes_):
            print(f"Original: '{label}' → Encoded: {i}")
    else:
        # Target is already numeric
        y_encoded = pre_model[target].values
    
    # Split the data
    X = pre_model.drop(target, axis=1)
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("\nTrain set class distribution:")
    print(Counter(y_train_encoded))
    
    X_balanced, y_balanced, weights = balance_dataset(X_train, y_train_encoded, method="over_under",
                                                      return_class_weights=True)
    
    # Create CatBoost pools with categorical features
    
    model = Model(problem_type="binary", weights=weights)
    model.initialize_model()
    
    train_pool = Pool(data=X_balanced, label=y_balanced, cat_features=cat_cols)
    test_pool = Pool(data=X_test, label=y_test_encoded, cat_features=cat_cols)
    model.fit(train_pool, test_pool, plot=True)
    
    y_pred = model.predict(test_pool)
    y_pred_proba = model.predict_proba(test_pool)[:, 1]
    
    print("\nClassification Report:")
    print(plot_confusion_matrix(y_test_encoded, y_pred))
    
  
    
    # Feature importance
    feature_importance = model.get_feature_importance(train_pool)
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values('Importance', ascending=True).reset_index(drop=True)
    print("\nTop 10 Important Features:")
    importance_df.head(5)
    
    # Get feature importances using the default method (PredictionValuesChange)
    feature_importance = model.get_feature_importance(train_pool)
    features = X_train.columns
    
    # Visualize the importances
    plt.figure(figsize=(10, 10))
    plt.barh(features, feature_importance)
    plt.xlabel("Importance")
    plt.title("CatBoost Feature Importance")
    plt.show()
    
    preds = model.predict(pre_model_future.drop('requested_bg', axis=1))
    
    conf_matrix = confusion_matrix(pre_model_future[target], label_encoder.inverse_transform(preds))
    print(classification_report(pre_model_future[target], label_encoder.inverse_transform(preds)))
    
    # Get unique class labels
    labels = np.unique(label_encoder.inverse_transform(preds))
    
    # Plot confusion matrix with labels
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()