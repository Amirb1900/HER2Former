import os
import json
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==============================
# Project Imports
# ==============================
from models.her2former import HER2Former
from her2_data.dataset import HER2Dataset
from her2_data.transforms import build_test_transforms


# ==============================
# Configuration
# ==============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CHECKPOINT_PATH = (
    "/kaggle/working/HER2Former/"
    "outputs/checkpoints/best_model.pth"
)


TEST_DIR = (
    "/kaggle/input/datasets/amirb1900/"
    "resized-bci-512/512BCI_dataset/"
    "test/test/HE"
)


OUTPUT_DIR = (
    "/kaggle/working/HER2Former/"
    "outputs/test_results"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


CLASS_NAMES = [
    "0",
    "1+",
    "2+",
    "3+"
]


BATCH_SIZE = 16


# ==============================
# Ordinal Decoder
# ==============================

def ordinal_logits_to_labels(logits):

    """
    CORAL decoding

    logits:
        [B,3]

    output:
        [B]
        classes 0-3
    """

    probs = torch.sigmoid(logits)

    labels = torch.sum(
        probs > 0.5,
        dim=1
    )

    return labels



# ==============================
# Load Model
# ==============================


def load_model():

    print("\nLoading model...")

    model = HER2Former(
        num_classes=4,
        top_k=64,
        freeze_backbone=False
    )


    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model.to(DEVICE)

    model.eval()


    print("Model loaded successfully")

    return model




# ==============================
# Dataset
# ==============================


def load_dataset():


    transform = build_test_transforms()


    dataset = HER2Dataset(
        root_dir=TEST_DIR,
        transform=transform
    )


    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    print(
        f"Test samples: {len(dataset)}"
    )


    return loader




# ==============================
# Evaluation
# ==============================


def evaluate(model, loader):


    print("\nRunning inference...")


    all_preds = []
    all_labels = []
    all_logits = []


    with torch.no_grad():


        for batch in tqdm(loader):


            images = batch[0]
            labels = batch[1]


            images = images.to(
                DEVICE
            )


            labels = labels.to(
                DEVICE
            )


            logits = model(
                images
            )


            preds = ordinal_logits_to_labels(
                logits
            )


            all_preds.extend(
                preds.cpu().numpy()
            )


            all_labels.extend(
                labels.cpu().numpy()
            )


            all_logits.extend(
                logits.cpu().numpy()
            )



    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_logits)
    )





# ==============================
# Save Results
# ==============================


def save_results(
        preds,
        labels,
        logits
):


    print("\n============================")
    print(" TEST RESULTS ")
    print("============================")


    acc = accuracy_score(
        labels,
        preds
    )


    macro_f1 = f1_score(
        labels,
        preds,
        average="macro"
    )


    weighted_f1 = f1_score(
        labels,
        preds,
        average="weighted"
    )


    qwk = cohen_kappa_score(
        labels,
        preds,
        weights="quadratic"
    )


    print(
        f"Accuracy : {acc:.4f}"
    )

    print(
        f"Macro F1 : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : {weighted_f1:.4f}"
    )

    print(
        f"QWK : {qwk:.4f}"
    )



    report = classification_report(
        labels,
        preds,
        target_names=CLASS_NAMES
    )


    print("\nClassification Report:")
    print(report)



    # JSON

    metrics = {

        "accuracy": float(acc),

        "macro_f1": float(macro_f1),

        "weighted_f1": float(weighted_f1),

        "qwk": float(qwk),

        "num_test_samples": int(len(labels))

    }


    with open(
        OUTPUT_DIR + "/metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )



    # report txt

    with open(
        OUTPUT_DIR +
        "/classification_report.txt",
        "w"
    ) as f:

        f.write(report)



    # predictions csv

    df = pd.DataFrame({

        "true_label": labels,

        "predicted_label": preds

    })


    df.to_csv(
        OUTPUT_DIR +
        "/predictions.csv",
        index=False
    )



    np.save(
        OUTPUT_DIR +
        "/predictions.npy",
        preds
    )



    # confusion matrix

    cm = confusion_matrix(
        labels,
        preds
    )


    plt.figure(
        figsize=(7,6)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )


    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.title(
        "HER2Former Confusion Matrix"
    )


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR +
        "/confusion_matrix.png",
        dpi=300
    )


    plt.close()



    print("\nSaved results:")
    print(OUTPUT_DIR)




# ==============================
# Main
# ==============================


if __name__ == "__main__":


    model = load_model()


    loader = load_dataset()


    preds, labels, logits = evaluate(
        model,
        loader
    )


    save_results(
        preds,
        labels,
        logits
    )
