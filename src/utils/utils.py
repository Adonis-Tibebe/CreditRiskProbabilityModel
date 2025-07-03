
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_test_split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Reusable function for splitting data into train and test sets.

    Parameters
    ----------
    X : array-like
        Features data.
    y : array-like
        Target data.
    test_size : float, optional
        Proportion of the data to include in the test chunk. By default, 0.2.
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls. By default, 42.
    stratify : array-like, optional
        If not None, data is split in a stratified fashion, using this as the class labels. By default, None.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Tuple containing train-test split of inputs.
    """


    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def evaluate_model(y_true, y_pred, y_proba, model_name="Model"):
    # Compute each metric
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    rocauc = roc_auc_score(y_true, y_proba)

    # Print nicely
    print(f"\n📌 ====== {model_name} ======")
    print(f"Accuracy: {acc:.4f} → Overall correct predictions.")
    print(f"Precision: {prec:.4f} → % of predicted positives actually positive.")
    print(f"Recall: {rec:.4f} → % of real positives captured.")
    print(f"F1 Score: {f1:.4f} → Balance of precision & recall.")
    print(f"ROC AUC: {rocauc:.4f} → Class separation quality.")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # ROC curve → shows tradeoff between TPR and FPR for different thresholds
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal = random chance
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend(loc="lower right")
    plt.show()

    # Return results for logging
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": rocauc
    }
