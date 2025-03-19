import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap


def plot_catboost_feature_importances(model, train_pool, n=10, figsize=(10, 10), file_name=None):
    """
    Retrieve, print, and plot feature importances from a CatBoost model using a given training Pool.

    Parameters:
        file_name:
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

    # Save or show the plot
    if file_name:
        plt.savefig(f"images/{file_name}.png", bbox_inches='tight')
        print(f"Confusion matrix saved to {file_name}")
    else:
        plt.show()

    return importance_df


def plot_catboost_shap(model, train_pool, X_train, max_display=25, figsize=(10, 10), save_path_beeswarm=None, save_path_bar=None):
    """
    Compute and plot SHAP values for a CatBoost model using a given training Pool.

    Parameters:
        model : CatBoost model
            A trained CatBoostClassifier or CatBoostRegressor.

        train_pool : Pool
            The CatBoost Pool used for training; it should include feature names.

        X_train : pd.DataFrame or np.ndarray
            The original feature matrix (including categorical features).

        max_display : int, default 10
            Maximum number of features to display in the bar plot summary.

        figsize : tuple, default (10, 10)
            Figure size for the plot.

        save_path_beeswarm : str, optional
            If provided, saves the beeswarm plot to the specified path.

        save_path_bar : str, optional
            If provided, saves the bar plot to the specified path.

    Returns:
        shap_values : np.ndarray
            The SHAP values used for plotting.
        importance_df : DataFrame
            A DataFrame summarizing mean absolute SHAP values per feature.
    """
    # Compute SHAP values using CatBoost's in-built functionality
    shap_values = model.get_feature_importance(train_pool, importance_type="ShapValues")

    # If the returned array has an extra column (expected value), remove it.
    if len(shap_values.shape) > 1 and shap_values.shape[1] == len(model.feature_names_) + 1:
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

    # Ensure X_train is a DataFrame with correct column names
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)

    # Generate and save both SHAP summary plots
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type='dot', show=False)
    if save_path_beeswarm:
        plt.savefig(save_path_beeswarm, bbox_inches='tight')
        print(f"Beeswarm plot saved to {save_path_beeswarm}")

    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type='bar', max_display=max_display, show=False)
    if save_path_bar:
        plt.savefig(save_path_bar, bbox_inches='tight')
        print(f"Bar plot saved to {save_path_bar}")

    # campaign_shap_values = shap_values[:, list(X_train.columns).index('feat_utm_campaign')]
    # # Combine SHAP values with original campaign data
    # campaign_df = pd.DataFrame({
    #     'campaign': X_train['feat_utm_campaign'],
    #     'shap_value': campaign_shap_values
    # })
    # campaign_df.to_csv("interim_datasets/Campaign_SHAP.tsv", sep="\t")
    #
    # campaign_shap_values = shap_values[:, list(X_train.columns).index('feat_in_contact_with_job_advisor')]
    # # Combine SHAP values with original campaign data
    # campaign_df = pd.DataFrame({
    #     'in_contact_with_job_advisor': X_train['feat_in_contact_with_job_advisor'],
    #     'shap_value': campaign_shap_values
    # })
    # campaign_df.to_csv("interim_datasets/in_contact_with_job_advisor_SHAP.tsv", sep="\t")

    return shap_values, importance_df

