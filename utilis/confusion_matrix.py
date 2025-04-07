import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_confusion_matrix(pred_labels_file, true_labels_file, output_file="confusion_matrix.png", class_names=None):
    """
    Load predicted labels and true labels, compute the confusion matrix, and visualize and save it.
    
    Parameters:
        pred_labels_file: Path to the .npy file containing predicted labels
        true_labels_file: Path to the .npy file containing true labels
        output_file: Path to save the confusion matrix image
        class_names: List of class names (optional), defaults to None and uses numeric indices if not provided
    """
    # Load predicted labels and true labels
    pred_labels = np.load(pred_labels_file)
    true_labels = np.load(true_labels_file)

    # Ensure the lengths of predicted labels and true labels match
    if len(pred_labels) != len(true_labels):
        raise ValueError(f"Length of predicted labels ({len(pred_labels)}) does not match length of true labels ({len(true_labels)})!")

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Use numeric indices if class_names is not provided
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(true_labels)))]

    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # Save confusion matrix image
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {output_file}")
    
    # Save confusion matrix values to a .npy file
    cm_npy_file = os.path.splitext(output_file)[0] + ".npy"
    np.save(cm_npy_file, cm)
    print(f"Confusion matrix values saved to {cm_npy_file}")

if __name__ == "__main__":
    # File paths
    pred_labels_file = "output/predicted_labels.npy"
    true_labels_file = "data/label/26.npy"
    
    class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]  
    
    # Generate and save confusion matrix
    plot_confusion_matrix(
        pred_labels_file=pred_labels_file,
        true_labels_file=true_labels_file,
        output_file="output/confusion_matrix.png",
        class_names=class_names
    )