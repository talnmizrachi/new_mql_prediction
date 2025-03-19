from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# First, encode your string labels to numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(pre_model[target])

# Check the mapping
for i, label in enumerate(label_encoder.classes_):
    print(f"Original: '{label}' → Encoded: {i}")

# Split the data using encoded labels
X_train, X_valid, y_train, y_valid = train_test_split(pre_model.drop(target, axis=1),
                                                      y_encoded,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      stratify=y_encoded
                                                      )

# Calculate class weights - use the actual class indices now
class_counts = np.bincount(y_encoded)
class_weights = {i: len(y_train) / (len(np.unique(y_encoded)) * count)
                 for i, count in enumerate(class_counts)}

# Create CatBoost Pools for training and validation
train_pool = Pool(X_train, y_train, cat_features=cat_cols)
test_pool = Pool(X_valid, y_valid, cat_features=cat_cols)


def catboost_eval(iterations, learning_rate, depth, l2_leaf_reg, random_strength, border_count, subsample, multiplier):
    """
    This function trains a CatBoost model with the given hyperparameters,
    and returns the F1 score on the validation set.
    """
    # Convert some parameters to integer values
    iterations = int(iterations)
    depth = int(depth)
    border_count = int(border_count)
    
    # Recalculate class weights with the multiplier for the positive class
    class_counts = np.bincount(y_train)
    class_weights = {i: len(y_train) / (len(np.unique(y_train)) * count)
                     for i, count in enumerate(class_counts)}
    class_weights[1] *= multiplier
    
    params = {
            'iterations': iterations,
            'learning_rate': learning_rate,
            'depth': depth,
            'l2_leaf_reg': l2_leaf_reg,
            'random_strength': random_strength,
            'subsample': subsample,
            'border_count': border_count,
            'loss_function': 'Logloss',
            'eval_metric': 'F1',
            "bootstrap_type": 'Bernoulli',
            'thread_count': 4,
            'od_type': 'Iter',
            'od_wait': int(iterations * 0.25),
            'random_seed': 42,
            'verbose': False,
            'class_weights': class_weights
    }
    
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)
    
    # Predict on the validation set (binary classification, so predictions are 0/1)
    preds = model.predict(test_pool)
    score = f1_score(y_valid, preds)
    return score


# Define the boundaries of the hyperparameters
pbounds = {
        'iterations': (200, 25000),
        'learning_rate': (0.0001, 0.35),
        'depth': (2, 12),
        'l2_leaf_reg': (1, 15),
        'random_strength': (1e-9, 15.0),
        'subsample': (0.0, 1.0),
        'border_count': (8, 255),
        'multiplier': (0.75, 5.5)
}

# Set up the Bayesian optimizer to maximize F1 score
optimizer = BayesianOptimization(
    f=catboost_eval,
    pbounds=pbounds,
    random_state=42,
)

# Run the optimizer: 2 random initial points, then 4 iterations
optimizer.maximize(
    init_points=75,
    n_iter=150,
)

# Display the best result found
print("Best result:", optimizer.max)

# Get the best score found during optimization
best_score = optimizer.max['target']
print(f"Best F1 score found: {best_score:.4f}")

# Get the parameters that produced this score
best_params = optimizer.max['params']
print("Parameters that achieved this score:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

# Get the best parameters from the optimizer
best_params = optimizer.max['params']
print("Best parameters found:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Convert integer parameters
iterations = int(best_params['iterations'])
depth = int(best_params['depth'])
border_count = int(best_params['border_count'])

# Calculate class weights with the optimized multiplier
class_counts = np.bincount(y_train)
class_weights = {i: len(y_train) / (len(np.unique(y_train)) * count)
                 for i, count in enumerate(class_counts)}
class_weights[1] *= best_params['multiplier']

# Create the final model with the best parameters
final_model = CatBoostClassifier(
    iterations=iterations,
    learning_rate=best_params['learning_rate'],
    depth=depth,
    l2_leaf_reg=best_params['l2_leaf_reg'],
    random_strength=best_params['random_strength'],
    border_count=border_count,
    subsample=best_params['subsample'],
    loss_function='Logloss',
    eval_metric='F1',
    custom_metric=['AUC', 'Precision'],
    bootstrap_type='Bernoulli',
    thread_count=4,
    random_seed=42,
    class_weights=class_weights,
    verbose=False,  # Set to False to avoid cluttering output
)

# Set up a plot for training visualization
plt.figure(figsize=(12, 8))

# Train the final model and capture the metrics for plotting
final_model.fit(
    train_pool,
    eval_set=test_pool,
    use_best_model=True,
    plot=True  # Enable CatBoost's built-in plotting
)

# Extract training history
train_metrics = final_model.get_evals_result()

# Plot learning curves
plt.figure(figsize=(15, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_metrics['learn']['Logloss'], label='Train Loss')
plt.plot(train_metrics['validation']['Logloss'], label='Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Logloss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# F1 score plot
plt.subplot(1, 2, 2)
plt.plot(train_metrics['validation']['F1'], label='Validation F1')
plt.xlabel('Iterations')
plt.ylabel('F1 Score')
plt.title('Validation F1 Score')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('catboost_learning_curves.png')
plt.show()

# Evaluate the model
train_preds = final_model.predict(train_pool)
train_f1 = f1_score(y_train, train_preds)
valid_preds = final_model.predict(test_pool)
valid_f1 = f1_score(y_valid, valid_preds)

print(f"Training F1 score: {train_f1:.4f}")
print(f"Validation F1 score: {valid_f1:.4f}")

# Get classification probabilities
valid_probs = final_model.predict_proba(test_pool)[:, 1]

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_valid, valid_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
# plt.savefig('catboost_confusion_matrix.png')
plt.show()

# Get detailed classification report
print("\nClassification Report (Validation):")
print(classification_report(y_valid, valid_preds))

# Plot ROC curve
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_valid, valid_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
# plt.savefig('catboost_roc_curve.png')
plt.show()

# Plot Precision-Recall curve
plt.figure(figsize=(10, 8))
precision, recall, _ = precision_recall_curve(y_valid, valid_probs)
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True, alpha=0.3)
# plt.savefig('catboost_precision_recall_curve.png')
plt.show()

# Feature Importance
plt.figure(figsize=(12, 10))
feature_importance = final_model.get_feature_importance(train_pool)
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(12, 8))
top_features = importance_df.head(20)
sns.barplot(x='Importance', y='Feature', data=top_features)
plt.title('Top 20 Feature Importance')
plt.tight_layout()
# plt.savefig('catboost_feature_importance.png')
plt.show()

# Threshold analysis
thresholds = np.arange(0.1, 1.0, 0.05)
f1_scores = []
precision_scores = []
recall_scores = []

for threshold in thresholds:
    preds_at_threshold = (valid_probs >= threshold).astype(int)
    f1 = f1_score(y_valid, preds_at_threshold)
    precision = precision_score(y_valid, preds_at_threshold)
    recall = recall_score(y_valid, preds_at_threshold)
    
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)

plt.figure(figsize=(12, 8))
plt.plot(thresholds, f1_scores, 'b-', label='F1 Score')
plt.plot(thresholds, precision_scores, 'g-', label='Precision')
plt.plot(thresholds, recall_scores, 'r-', label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Metrics at Different Thresholds')
plt.legend()
plt.grid(True, alpha=0.3)
# plt.savefig('catboost_threshold_analysis.png')
plt.show()

# Save the model for later use
# final_model.save_model("catboost_optimized_model.cbm")
