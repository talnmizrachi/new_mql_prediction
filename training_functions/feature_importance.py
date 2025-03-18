import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap


def plot_catboost_feature_importances(model, train_pool, n=10, figsize=(10, 10)):
    """
    Retrieve, print, and plot feature importances from a CatBoost model using a given training Pool.

    Parameters:
        model : CatBoost model
            The trained CatBoost model.

        train_pool : Pool
            The CatBoost Pool used for training; must include feature names.

        top_n : int, default=10
            Number of top important features to display.

        bottom_n : int, default=5
            Number of least important features to display.

        figsize : tuple, default=(10, 10)
            Figure size for the importance plot.

    Returns:
        importance_df : DataFrame
            DataFrame of features and their importances sorted in ascending order.
    """
    # Retrieve feature importances and names from the model
    feature_importances = model.get_feature_importance(train_pool)
    feature_names = model.feature_names_
    
    # Create a DataFrame for easy inspection
    importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
    })
    importance_df = importance_df.sort_values('Importance', ascending=True).reset_index(drop=True)
    
    # Print the top and bottom important features
    print("\nTop {} Important Features:".format(n))
    print(importance_df.tail(n))
    
    print("\nBottom {} Important Features:".format(n))
    print(importance_df.head(n))
    
    # Visualize the feature importances
    plt.figure(figsize=figsize)
    plt.barh(feature_names, feature_importances)
    plt.xlabel("Importance")
    plt.title("CatBoost Feature Importance")
    plt.show()
    
    return importance_df





def plot_catboost_shap(model, train_pool, plot_type='beeswarm', max_display=10, figsize=(10, 10)):
    """
    Compute and plot SHAP values for a CatBoost model using a given training Pool.

    Parameters:
        model : CatBoost model
            A trained CatBoostClassifier or CatBoostRegressor.

        train_pool : Pool
            The CatBoost Pool used for training; it should include feature names.

        plot_type : str, default 'beeswarm'
            Type of SHAP plot to display. Options:
                - 'beeswarm': Displays the typical SHAP beeswarm plot.
                - 'bar': Displays a bar plot summarizing the mean absolute SHAP values.

        max_display : int, default 10
            Maximum number of features to display in the bar plot summary.

        figsize : tuple, default (10, 10)
            Figure size for the plot.

    Returns:
        shap_values : np.ndarray
            The SHAP values used for plotting.
        importance_df : DataFrame
            A DataFrame summarizing mean absolute SHAP values per feature.
    """
    # Compute SHAP values using CatBoost's in-built functionality
    shap_values = model.get_feature_importance(train_pool, type="ShapValues")
    
    # If the returned array has an extra column (expected value), remove it.
    if shap_values.shape[1] == len(model.feature_names_) + 1:
        shap_values = shap_values[:, :-1]
    
    # Get feature names directly from the model
    feature_names = model.feature_names_
    
    # Compute mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
            'Feature': feature_names,
            'MeanAbsShap': mean_abs_shap
    }).sort_values('MeanAbsShap', ascending=False).reset_index(drop=True)
    
    print(f"\nTop {max_display} Features by Mean Absolute SHAP Value:")
    print(importance_df.head(max_display))
    
    # Extract the features from the pool; if not a DataFrame, create one.
    X_train = train_pool.get_features()
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)
    
    # Generate the SHAP summary plot
    plt.figure(figsize=figsize)
    if plot_type == 'beeswarm':
        shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type='dot', show=True)
    elif plot_type == 'bar':
        shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type='bar', max_display=max_display,
                          show=True)
    else:
        raise ValueError("Invalid plot_type. Choose 'beeswarm' or 'bar'.")
    
    return shap_values, importance_df
