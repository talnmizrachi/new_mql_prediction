from catboost import CatBoostClassifier, CatBoostRegressor, Pool

# Create CatBoost pools with categorical features
train_pool = Pool(X_balanced, y_balanced, cat_features=categorical_cols)
test_pool = Pool(X_test, y_test_encoded, cat_features=categorical_cols)


class Model:
    def __init__(self, type='binary'):
        self.type = type
        
    def init_model(self, weights=None):
        if self.type == "regression":
            ...
        if self.type =="binary":
            model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.02,
                depth=3,
                l2_leaf_reg=12,
                bootstrap_type='Bernoulli',
                subsample=0.35,
                sampling_frequency='PerTreeLevel',
                eval_metric='Precision',  # Adjust evaluation metric as needed
                custom_metric=['F1', 'AUC'],
                loss_function='Logloss',  # Binary classification loss function
                class_weights=weights,  # Only needed for classification tasks
                use_best_model=True,
                random_seed=42,
                verbose=250,
                random_strength=5.15,
                od_type='Iter',
                od_wait=500,
            )
        if self.type == "multiclass":
            model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.02,
                depth=3,
                l2_leaf_reg=12,
                bootstrap_type='Bernoulli',
                subsample=0.35,
                sampling_frequency='PerTreeLevel',
                eval_metric='Accuracy',  # Or another metric suitable for multiclass tasks
                loss_function='MultiClass',  # For multiclass classification
                use_best_model=True,
                random_seed=42,
                verbose=250,
                random_strength=5.15,
                od_type='Iter',
                od_wait=500,
            )
            return model
        raise ValueError("Unsupported problem type. Choose 'binary', 'multiclass', or 'regression'.")
        
model = Model(type="binary")
# Fit the model using the pools; plot=True displays training progress
model.fit(train_pool, eval_set=test_pool, plot=True, use_best_model=True)
