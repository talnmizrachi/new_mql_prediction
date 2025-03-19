import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, label_encoder=None, figsize=(10, 6), cmap='Blues',
                          title='Confusion Matrix', vmax=None, file_name=None):
    """
    Plot a confusion matrix for binary or multiclass classification.

    Parameters:
        y_true: array-like of shape (n_samples,)
            True labels.
        y_pred: array-like of shape (n_samples,)
            Predicted labels.
        label_encoder: optional
            A fitted label encoder with an inverse_transform method.
            If provided, will be used to decode the predicted labels for display.
        figsize: tuple, default (10, 6)
            Figure size for the plot.
        cmap: str, default 'Blues'
            Colormap for the heatmap.
        title: str, default 'Confusion Matrix'
            Title for the plot.
        vmax: float, optional
            Maximum value for the color scale. If None, defaults to the maximum value in the confusion matrix.
        file_name: str, optional
            If provided, saves the figure to the given path. Otherwise, displays the plot.

    Returns:
        None
    """
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Determine display labels: use the label encoder if provided, otherwise use the union of y_true and y_pred
    if label_encoder is not None:
        labels = np.unique(label_encoder.inverse_transform(y_pred))
    else:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    # Set vmax to the maximum value in the matrix if not provided
    if vmax is None:
        vmax = conf_matrix.max()

    # Plot the confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels, vmax=vmax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)

    # Save or show the plot
    if file_name:
        plt.savefig(f"images/{file_name}.png", bbox_inches='tight')
        print(f"Confusion matrix saved to {file_name}")
    else:
        plt.show()