from catboost import CatBoostClassifier, CatBoostRegressor, Pool


class Model:
    def __init__(self, problem_type='binary', weights=None, **kwargs):
        """
        Initializes the Model class.

        Parameters:
            problem_type (str): 'binary', 'multiclass', or 'regression'
            weights: Class weights for classification tasks (if applicable)
            **kwargs: Additional hyperparameters to override defaults
        """
        self.problem_type = problem_type
        self.weights = weights
        self.is_init = False
        self.model = None
        # Store any additional hyperparameter overrides
        self.extra_params = kwargs
    
    def initialize_model(self):
        """
        Initializes and returns a CatBoost model based on the problem type.
        """
        # Common hyperparameters for all models
        common_params = {
                'iterations': 1000,
                'learning_rate': 0.02,
                'depth': 3,
                'l2_leaf_reg': 12,
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.35,
                'sampling_frequency': 'PerTreeLevel',
                'use_best_model': True,
                'random_seed': 42,
                'verbose': 250,
                'random_strength': 5.15,
                'od_type': 'Iter',
                'od_wait': 500
        }
        # Update common parameters with any extra overrides
        common_params.update(self.extra_params)
        
        # Create model based on problem type
        if self.problem_type == "regression":
            self.model = CatBoostRegressor(
                **common_params,
                eval_metric='RMSE',  # Regression evaluation metric
                loss_function='RMSE'  # Regression loss function
            )
        elif self.problem_type == "binary":
            self.model = CatBoostClassifier(
                **common_params,
                eval_metric='Precision',  # Classification evaluation metric
                custom_metric=['F1', 'AUC'],
                loss_function='Logloss',  # Binary classification loss function
                class_weights=self.weights  # Optional: class weights for imbalance
            )
        elif self.problem_type == "multiclass":
            self.model = CatBoostClassifier(
                **common_params,
                eval_metric='Accuracy',  # Multiclass evaluation metric
                loss_function='MultiClass'  # Multiclass loss function
            )
        else:
            raise ValueError("Unsupported problem type. Choose 'binary', 'multiclass', or 'regression'.")
        
        self.is_init = True
        return self.model
    
    def fit(self, _train_pool, _test_pool, plot=False):
        """
        Fits the initialized model on the training pool and evaluates on the test pool.

        Parameters:
            _train_pool: CatBoost Pool for training
            _test_pool: CatBoost Pool for evaluation
            plot (bool): Whether to plot training progress.
        """
        if not self.is_init or self.model is None:
            raise ValueError("Model is not initialized. Call initialize_model() before fit().")
        self.model.fit(_train_pool, eval_set=_test_pool, plot=plot, use_best_model=True)
    
    def predict(self, X):
        """
        Makes predictions using the trained model.

        Parameters:
            X: Features for which to predict.

        Returns:
            Predictions from the model.
        """
        if not self.is_init or self.model is None:
            raise ValueError("Model is not initialized. Call initialize_model() before predict().")
        return self.model.predict(X)
        

if __name__ == '__main__':
    # Create CatBoost pools with categorical features
    train_pool = Pool(X_balanced, y_balanced, cat_features=categorical_cols)
    test_pool = Pool(X_test, y_test_encoded, cat_features=categorical_cols)
    model = Model(type="binary")
    # Fit the model using the pools; plot=True displays training progress
    model.fit(train_pool, eval_set=test_pool, plot=True, use_best_model=True)
