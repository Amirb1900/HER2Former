import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    cohen_kappa_score,
    f1_score
)

from torch.utils.data import DataLoader


# ==========================
# Project imports
# ==========================

from models.her2former import HER2Former
from her2_data.dataset import HER2Dataset
from her2_data.transforms import build_test_transforms



# ==========================
# Config
# ==========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CHECKPOINT = (
    "/kaggle/working/HER2Former/"
    "outputs/checkpoints/best_model.pth"
)


TEST_DIR = (
    "/kaggle/input/datasets/amirb1900/"
    "resized-bci-512/512BCI_dataset/"
    "test/test/HE"
)


BATCH_SIZE = 16


CLASS_NAMES = [
    "0",
    "1+",
    "2+",
    "3+"
]



# ==========================
# Ordinal conversion
# ==========================

def ordinal_logits_to_labels(logits):

    """
    logits:
    [B,3]

    return:
    labels:
    [B]

    """

    probs = torch.sigmoid(logits)

    labels = torch.sum(
        probs > 0.5,
        dim=1
    )

    return labels



# ==========================
# Load model
# ==========================


print("\nLoading model...")


model = HER2Former(
    num_classes=4,
    top_k=64,
    freeze_backbone=False
)


checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.to(DEVICE)

model.eval()


print("Model loaded successfully")



# ==========================
# Dataset
# ==========================


test_dataset = HER2Dataset(
    root_dir=TEST_DIR,
    transform=build_test_transforms()
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


print(
    "Test samples:",
    len(test_dataset)
)



# ==========================
# Prediction
# ==========================


all_preds=[]
all_labels=[]


print("\nRunning inference...")


with torch.no_grad():

    for batch in tqdm(test_loader):

        images = batch[0]
        labels = batch[1]


        images = images.to(DEVICE)


        outputs = model(images)


        preds = ordinal_logits_to_labels(
            outputs
        )


        all_preds.extend(
            preds.cpu().numpy()
        )


        all_labels.extend(
            labels.numpy()
        )



# ==========================
# Metrics
# ==========================


all_preds=np.array(all_preds)
all_labels=np.array(all_labels)



acc = accuracy_score(
    all_labels,
    all_preds
)


f1 = f1_score(
    all_labels,
    all_preds,
    average="macro"
)


qwk = cohen_kappa_score(
    all_labels,
    all_preds,
    weights="quadratic"
)



print("\n============================")
print(" TEST RESULTS ")
print("============================")


print(
    f"Accuracy : {acc:.4f}"
)


print(
    f"Macro F1 : {f1:.4f}"
)


print(
    f"QWK      : {qwk:.4f}"
)



print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES
    )
)



# ==========================
# Confusion Matrix
# ==========================


cm = confusion_matrix(
    all_labels,
    all_preds
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)


disp.plot()

plt.title(
    "HER2Former Confusion Matrix"
)

plt.show()



# ==========================
# Save predictions
# ==========================


save_path = (
    "/kaggle/working/HER2Former/"
    "outputs/test_predictions.npy"
)


np.save(
    save_path,
    {
        "labels":all_labels,
        "predictions":all_preds
    }
)


print(
    "\nPredictions saved:"
)

print(save_path)
