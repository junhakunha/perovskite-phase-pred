import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



def create_confusion_matrix(model, test_dataloader, save_path=None):
    all_outputs = []
    all_classes = []
    with torch.no_grad():
        for inputs, classes in test_dataloader:
            outputs = model(inputs)
            all_outputs.append(outputs)
            all_classes.append(classes)
    all_outputs = torch.round(torch.cat(all_outputs)).flatten()
    all_classes = torch.round(torch.cat(all_classes)).flatten()

    actual_labels = all_classes.tolist()
    predicted_labels = all_outputs.tolist()

    # Create confusion matrix
    cm = confusion_matrix(actual_labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(
        actual_labels), yticklabels=np.unique(actual_labels))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()