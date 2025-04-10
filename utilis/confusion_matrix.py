import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_combined_confusion_matrix(pred_files_dict, output_file="combined_confusion_matrix.png", class_names=None):
    """
    Load multiple predicted and true label files, combine them to calculate the confusion matrix, and visualize and save it.
    
    Parameters:
        pred_files_dict: Dictionary with predicted label file paths as keys and corresponding true label file paths as values
        output_file: Path to save the confusion matrix image
        class_names: List of class names (optional), defaults to None and uses numeric indices if not provided
    """
    # Initialize lists to store all predicted and true labels
    all_pred_labels = []
    all_true_labels = []
    
    # Iterate through all file pairs
    for pred_file, true_file in pred_files_dict.items():
        # Load predicted and true labels
        pred_labels = np.load(pred_file)
        true_labels = np.load(true_file)
        
        # Ensure the length of predicted and true labels matches for each file pair
        if len(pred_labels) != len(true_labels):
            raise ValueError(f"The length of predicted labels in file {pred_file} ({len(pred_labels)}) "
                           f"does not match the length of true labels in {true_file} ({len(true_labels)})!")
        
        # Add to the total lists
        all_pred_labels.extend(pred_labels)
        all_true_labels.extend(true_labels)
    
    # Convert to numpy arrays
    all_pred_labels = np.array(all_pred_labels)
    all_true_labels = np.array(all_true_labels)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    
    # If class_names is not provided, use numeric indices
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(all_true_labels)))]
    
    # Visualize the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # Save the confusion matrix image
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix has been saved to {output_file}")
    
    # Save the confusion matrix values to a .npy file
    cm_npy_file = os.path.splitext(output_file)[0] + ".npy"
    np.save(cm_npy_file, cm)
    print(f"Confusion matrix values have been saved to {cm_npy_file}")

if __name__ == "__main__":
    # Define the file path dictionary
    pred_files_dict = {
        "output/predicted_labels_26.npy": "data/label/26.npy",
        "output/predicted_labels_27.npy": "data/label/27.npy",
        "output/predicted_labels_28.npy": "data/label/28.npy",
        "output/predicted_labels_29.npy": "data/label/29.npy",
        "output/predicted_labels_30.npy": "data/label/30.npy"
    }
    
    # Define class names
    class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]  
    
    # Generate and save the combined confusion matrix
    plot_combined_confusion_matrix(
        pred_files_dict=pred_files_dict,
        output_file="output/confusion_matrix.png",
        class_names=class_names
    )