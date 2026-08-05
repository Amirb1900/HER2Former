import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    cohen_kappa_score
)

import seaborn as sns


# ==============================
# Paths
# ==============================

PRED_PATH = "/kaggle/working/HER2Former/outputs/test_predictions.npy"

OUT_DIR = "/kaggle/working/HER2Former/outputs/test_analysis"

os.makedirs(OUT_DIR, exist_ok=True)


# ==============================
# Load Predictions
# ==============================

print("\nLoading predictions...")

data = np.load(
    PRED_PATH,
    allow_pickle=True
).item()


labels = data["labels"]
predictions = data["predictions"]


print("Samples:", len(labels))
print("Labels shape:", labels.shape)
print("Predictions shape:", predictions.shape)



# ==============================
# Metrics
# ==============================

accuracy = accuracy_score(
    labels,
    predictions
)

macro_f1 = f1_score(
    labels,
    predictions,
    average="macro"
)

weighted_f1 = f1_score(
    labels,
    predictions,
    average="weighted"
)

qwk = cohen_kappa_score(
    labels,
    predictions,
    weights="quadratic"
)



print("\n==============================")
print(" FINAL TEST ANALYSIS ")
print("==============================")

print(f"Accuracy     : {accuracy:.4f}")
print(f"Macro F1     : {macro_f1:.4f}")
print(f"Weighted F1  : {weighted_f1:.4f}")
print(f"QWK          : {qwk:.4f}")



# ==============================
# Classification Report
# ==============================

classes = [
    "0",
    "1+",
    "2+",
    "3+"
]


report = classification_report(
    labels,
    predictions,
    target_names=classes
)


print("\nClassification Report\n")
print(report)



with open(
    os.path.join(
        OUT_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)



# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(
    labels,
    predictions
)


print("\nConfusion Matrix:")
print(cm)



plt.figure(
    figsize=(7,6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes
)


plt.xlabel(
    "Prediction"
)

plt.ylabel(
    "Ground Truth"
)

plt.title(
    "HER2Former Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)


plt.close()



# ==============================
# Normalized Confusion Matrix
# ==============================


cm_norm = cm.astype(float) / cm.sum(axis=1)[:,None]


plt.figure(
    figsize=(7,6)
)


sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    xticklabels=classes,
    yticklabels=classes
)


plt.xlabel(
    "Prediction"
)

plt.ylabel(
    "Ground Truth"
)

plt.title(
    "Normalized Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUT_DIR,
        "normalized_confusion_matrix.png"
    ),
    dpi=300
)


plt.close()



# ==============================
# Class Distribution
# ==============================


plt.figure(
    figsize=(7,5)
)


plt.hist(
    labels,
    bins=4,
    rwidth=0.8
)


plt.xticks(
    range(4),
    classes
)


plt.title(
    "Ground Truth Distribution"
)


plt.xlabel(
    "HER2 Grade"
)


plt.ylabel(
    "Count"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUT_DIR,
        "ground_truth_distribution.png"
    ),
    dpi=300
)


plt.close()



# ==============================
# Prediction Distribution
# ==============================


plt.figure(
    figsize=(7,5)
)


plt.hist(
    predictions,
    bins=4,
    rwidth=0.8
)


plt.xticks(
    range(4),
    classes
)


plt.title(
    "Prediction Distribution"
)


plt.xlabel(
    "HER2 Grade"
)


plt.ylabel(
    "Count"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUT_DIR,
        "prediction_distribution.png"
    ),
    dpi=300
)


plt.close()



# ==============================
# Save Summary
# ==============================


summary = f"""
HER2Former Test Analysis

Samples:
{len(labels)}

Accuracy:
{accuracy:.4f}

Macro F1:
{macro_f1:.4f}

Weighted F1:
{weighted_f1:.4f}

Quadratic Weighted Kappa:
{qwk:.4f}

"""


with open(
    os.path.join(
        OUT_DIR,
        "summary.txt"
    ),
    "w"
) as f:

    f.write(summary)



print("\n==============================")
print("Analysis Finished")
print("==============================")

print(
    "Results saved in:"
)

print(
    OUT_DIR
)
