import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to the results CSV files and output folders
DISEASES = {
    "DR": {
        "csv": os.path.join(SCRIPT_DIR, "DR", "MC_results_DR_binary.csv"),
        "folder": os.path.join(SCRIPT_DIR, "DR"),
        "labels": ['No DR', 'DR']
    },
    "BT": {
        "csv": os.path.join(SCRIPT_DIR, "BT", "MC_results_BT.csv"),
        "folder": os.path.join(SCRIPT_DIR, "BT"),
        "labels": ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    },
    "BC": {
        "csv": os.path.join(SCRIPT_DIR, "BC", "MC_results_BC.csv"),
        "folder": os.path.join(SCRIPT_DIR, "BC"),
        "labels": ['Non-IDC', 'IDC']
    },
    "CX": {
        "csv": os.path.join(SCRIPT_DIR, "CX", "MC_results_CX_binary.csv"),
        "folder": os.path.join(SCRIPT_DIR, "CX"),
        "labels": ['Normal', 'Abnormal']
    }
}

def main():
    for name, info in DISEASES.items():
        csv_path = info["csv"]
        folder = info["folder"]
        display_labels = info["labels"]
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # 1. Box Plot for Uncertainty (Individual)
            plt.figure(figsize=(6, 5))
            sns.boxplot(y=df['Uncertainty'], color='skyblue')
            plt.title(f'Uncertainty Distribution: {name}')
            plt.ylabel('Uncertainty')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            box_plot_path = os.path.join(folder, f"uncertainty_boxplot_{name}.png")
            plt.savefig(box_plot_path, dpi=150)
            plt.close()
            print(f"Saved Box Plot for {name} to {box_plot_path}")
            
            # 2. Confusion Matrix (Individual)
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(df['Label'], df['Prediction'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            disp.plot(cmap='Blues', colorbar=True)
            plt.title(f'Confusion Matrix: {name}')
            cm_plot_path = os.path.join(folder, f"confusion_matrix_{name}.png")
            plt.savefig(cm_plot_path, dpi=150)
            plt.close()
            print(f"Saved Confusion Matrix for {name} to {cm_plot_path}")
        else:
            print(f"Warning: CSV not found for {name} at {csv_path}")

if __name__ == "__main__":
    main()
