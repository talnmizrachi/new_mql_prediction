import warnings
# from training_functions.feature_importance import plot_catboost_feature_importances, plot_catboost_shap
from minus_ones import minus_one_dict
from training_functions.balancing_data import balance_dataset
from training_functions.classifiers import Model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from catboost import Pool
from sklearn.model_selection import train_test_split
from training_functions.visuals import plot_confusion_matrix
from collections import Counter
import numpy as np
from training_functions.funnel_dictionary import get_sorted_funnel_for_
from clearml import Task
from sklearn.metrics import precision_score, recall_score

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)

# Initialize ClearML task
task = Task.init(project_name="my project", task_name="first_run")


def evaluate_model(y_test, y_prediction):
    precision = precision_score(y_test, y_prediction, pos_label=1)
    recall = recall_score(y_test, y_prediction, pos_label=1)
    print(f"Precision (Minority Class): {precision:.4f}")
    print(f"Recall (Minority Class): {recall:.4f}")
    return precision, recall


def read_and_prepare_for_training(starting_level):
    _df = pd.read_csv('training_data/preprocessed_50.tsv', sep="\t", index_col='id')
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


def split_to_current_and_future(_df, _cutoff, _start_date):
    # Adjust the filtering conditions as needed.
    conditions = _df['in_contact_with_job_advisor'].notna()
    _future_test = _df[(_df['mql_date'] >= _cutoff) & conditions].copy()
    _current_df = _df[(_df['mql_date'] < _cutoff) & (_df['mql_date'] >= _start_date) & conditions].copy()
    return _current_df, _future_test


def extract_columns_data(_df, _starting_level, _target):
    features_ = [col for col in _df.columns if col.startswith("feat")]
    text_features_ = [col for col in _df.columns if col.startswith("svd")]
    all_features = text_features_ + features_
    non_feature_col = [col for col in _df.columns if col not in all_features]
    cat_cols, numerical = get_sorted_funnel_for_(_starting_level, text_features_)
    final_columns = numerical + cat_cols + [_target]
    return features_, text_features_, all_features, non_feature_col, cat_cols, numerical, final_columns


if __name__ == '__main__':
    cutoff = '2025-02-01'
    start_date = "2023-01-01"
    starting_level = "mql"
    target = 'enrolled'
    # Initial winner code as an identifier
    winner_code = "c7ad10efb9314a3095962deb12b7171d"
    max_precision, max_recall = -100, -100
    
    df = read_and_prepare_for_training(starting_level)
    features, text_features, all_features, non_features, cat_cols, numeric_col, final_columns = extract_columns_data(df,
                                                                                                                     starting_level,
                                                                                                                     target)
    
    # Filter rows that have at least 10% of non-feature columns not null
    df = df[(df[non_features].notna().sum(axis=1) / len(df.columns)) > 0.1].copy()
    
    df['is_requested'] = df['requested_bg'].eq('requested')
    df['enrolled'] = np.where(df['closed_won_deal__program_duration'].notna(), "enrolled", "didn't enroll")
    
    # Convert categorical columns to string
    for col in cat_cols:
        df[col] = df[col].astype(str)
    
    # Split the dataset based on dates
    current_df, future_test = split_to_current_and_future(df, cutoff, start_date)
    
    for code, columns in minus_one_dict().items():
        
        new_num_cols = list(set(numeric_col).intersection(set(columns))) + text_features
        new_cat_cols = list(set(cat_cols).intersection(set(columns)))
        
        print(f"numeric columns for this iteration: {new_num_cols}")
        print(f"categorical columns for this iteration: {new_cat_cols}")
        
        # Create a fresh list of final columns based on this subset
        current_final_columns = list(columns) + [target]
        pre_model = current_df.copy()[current_final_columns].copy()
        pre_model_future = future_test.copy()[current_final_columns].copy()
        
        print(len(pre_model), list(pre_model[target].value_counts().items()))
        print(len(pre_model_future), list(pre_model_future[target].value_counts().items()))
        
        print("Original class distribution:")
        print(pre_model[target].value_counts())
        counts = pre_model[target].value_counts()
        if len(counts) >= 2 and counts.iloc[1] != 0:
            imbalance_ratio = counts.iloc[0] / counts.iloc[1]
            print(f"Imbalance ratio: 1:{imbalance_ratio:.2f}")
        else:
            print("Imbalance ratio: Not available (one class might be missing)")
        
        # Create local copies to avoid unwanted side effects
        current_cat_cols = cat_cols.copy()
        current_numeric_cols = numeric_col.copy()
        
        if target in current_cat_cols:
            current_cat_cols.remove(target)
        if target in current_numeric_cols:
            current_numeric_cols.remove(target)
        
        print(f"\nCategorical columns: {len(current_cat_cols)}")
        print(f"Numerical columns: {len(current_numeric_cols)}")
        
        # Encode target if needed
        if pre_model[target].dtype in ('object', "category", "string"):
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(pre_model[target])
            for i, label in enumerate(label_encoder.classes_):
                print(f"Original: '{label}' → Encoded: {i}")
        else:
            y_encoded = pre_model[target].values
        
        # Split the data
        X = pre_model.drop(target, axis=1)
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        
        print("\nTrain set class distribution:")
        print(Counter(y_train_encoded))
        
        X_balanced, y_balanced, weights = balance_dataset(
            X_train, y_train_encoded, method="over_under", return_class_weights=True)
        
        print(f"Weights: {weights}")
        task.connect({"weights_dict": weights})
        print(f"Punished Weights: {weights}")
        
        model = Model(problem_type="binary", weights=weights)
        model.initialize_model()
        task.connect(model.hyperparams)
        
        train_pool = Pool(data=X_balanced, label=y_balanced, cat_features=new_cat_cols)
        test_pool = Pool(data=X_test, label=y_test_encoded, cat_features=new_cat_cols)
        model.fit(train_pool, test_pool, plot=True)
        
        y_pred = model.predict(test_pool)
        print("\nClassification Report:")
        print(plot_confusion_matrix(y_test_encoded, y_pred, file_name="confusion_matrix"))
        precision, recall = evaluate_model(y_test_encoded, y_pred)
        
        # Update the winner if the current model improves
        if precision >= max_precision and recall >= max_recall and (precision != max_precision or recall != max_recall):
            max_precision = precision
            max_recall = recall
            task.connect({"precision": precision, "recall": recall, "code": code})
            print("Winning columns:", columns)
            if code == "c7ad10efb9314a3095962deb12b7171d":
                task.add_tags(["all_cols"])
            task.add_tags(["Winner"])
            # Rename the current task to include the experiment code
            task.set_name(f"Experiment with columns: {code}")
            task.close()  # Close the current task
            
            # Initialize a new task with a temporary name for the next experiment
            task = Task.init(project_name="my project", task_name="temporary_task_name")
            winner_code = code
        
        # plot_catboost_shap(
        #     model,
        #     train_pool,
        #     X_balanced,
        #     save_path_beeswarm='images/beeswarm_plot.png',
        #     save_path_bar='images/bar_plot.png'
        # )
        # plot_catboost_feature_importances(model, train_pool,  file_name="feature_importance")
        print(1)
        # preds = model.predict(pre_model_future.drop('requested_bg', axis=1))
        # conf_matrix = confusion_matrix(pre_model_future[target], label_encoder.inverse_transform(preds))
        # print(classification_report(pre_model_future[target], label_encoder.inverse_transform(preds)))
        #
        # # Get unique class labels
        # labels = np.unique(label_encoder.inverse_transform(preds))
        #
        # # Plot confusion matrix with labels
        # plt.figure(figsize=(10, 6))
        # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')
        # plt.title('Confusion Matrix')
        # plt.show()
