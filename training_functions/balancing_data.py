from collections import Counter
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def balance_dataset(X_train, y_train,
                    method='over_under',
                    sampling_strategy_over=0.7,
                    sampling_strategy_under=0.8,
                    multiclass=False,
                    return_class_weights=False):
    """
    Balance a dataset using oversampling, undersampling, or a combination of both.

    Parameters:
        X_train, y_train : array-like
            Training data and labels.

        method : str, default 'over_under'
            The balancing method. Choose one of:
              - 'over'       : Apply RandomOverSampler only.
              - 'under'      : Apply RandomUnderSampler only.
              - 'over_under' : Apply oversampling first then undersampling.

        sampling_strategy_over : float, default 0.7
            The sampling strategy for RandomOverSampler. For binary tasks, this is a ratio.
            For multiclass tasks, use a float similarly to balance minority classes.

        sampling_strategy_under : float or dict, default 0.8
            The sampling strategy for RandomUnderSampler. For binary tasks, a float represents
            the desired ratio (minority / majority) after resampling.
            For multiclass tasks, if provided as a float, each class with more samples than the smallest
            is undersampled to not exceed int(min_count / sampling_strategy_under). Alternatively,
            you can provide a dictionary mapping class labels to desired sample counts.

        multiclass : bool, default False
            Set to True if dealing with a multiclass problem.

        return_class_weights : bool, default False
            If True, also return class weights computed inversely proportional to class frequencies.

    Returns:
        If return_class_weights is False:
            X_balanced, y_balanced

        If return_class_weights is True:
            X_balanced, y_balanced, class_weights
    """
    
    if method not in ['over', 'under', 'over_under']:
        raise ValueError("Invalid method specified. Choose from 'over', 'under', or 'over_under'")
    
    # ---------------------
    # Oversampling only
    # ---------------------
    if method == 'over':
        over = RandomOverSampler(sampling_strategy=sampling_strategy_over, random_state=42)
        X_balanced, y_balanced = over.fit_resample(X_train, y_train)
    
    # ---------------------
    # Undersampling only
    # ---------------------
    elif method == 'under':
        if multiclass and isinstance(sampling_strategy_under, float):
            counts = Counter(y_train)
            min_count = min(counts.values())
            strategy_dict = {}
            for cls, count in counts.items():
                # For classes with more samples than the minimum,
                # target count = int(min_count / sampling_strategy_under)
                if count > min_count:
                    new_count = int(min_count / sampling_strategy_under)
                    strategy_dict[cls] = min(new_count, count)
                else:
                    strategy_dict[cls] = count
            sampling_strategy_val = strategy_dict
        else:
            sampling_strategy_val = sampling_strategy_under
        
        under = RandomUnderSampler(sampling_strategy=sampling_strategy_val, random_state=42)
        X_balanced, y_balanced = under.fit_resample(X_train, y_train)
    
    # ---------------------
    # Combined oversampling then undersampling
    # ---------------------
    elif method == 'over_under':
        # Step 1: Oversample the minority class
        over = RandomOverSampler(sampling_strategy=sampling_strategy_over, random_state=42)
        X_over, y_over = over.fit_resample(X_train, y_train)
        
        # Step 2: Undersample the majority class
        if multiclass and isinstance(sampling_strategy_under, float):
            counts = Counter(y_over)
            min_count = min(counts.values())
            strategy_dict = {}
            for cls, count in counts.items():
                if count > min_count:
                    new_count = int(min_count / sampling_strategy_under)
                    strategy_dict[cls] = min(new_count, count)
                else:
                    strategy_dict[cls] = count
            sampling_strategy_val = strategy_dict
        else:
            sampling_strategy_val = sampling_strategy_under
        
        under = RandomUnderSampler(sampling_strategy=sampling_strategy_val, random_state=42)
        X_balanced, y_balanced = under.fit_resample(X_over, y_over)
    
    # ---------------------
    # Optionally, compute class weights
    # ---------------------
    if return_class_weights:
        # Ensure y_balanced is a numpy array for bincount
        y_arr = np.array(y_balanced)
        class_counts = np.bincount(y_arr)
        total_samples = len(y_arr)
        class_weights = {i: total_samples / (len(class_counts) * count)
                         for i, count in enumerate(class_counts)}
        return X_balanced, y_balanced, class_weights
    
    return X_balanced, y_balanced
