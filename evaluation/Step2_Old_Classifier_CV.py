import os
import csv
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
BE_PROJECT = os.path.abspath(os.path.join(HERE, ".."))
DATASET = os.path.join(BE_PROJECT, "Dataset")
DATA_DIR = os.path.join(DATASET, "Generated_FAF_CNN_Dataset_Refactored_3classes")
GROUPS_FILE = os.path.join(DATASET, "Splits", "patient_groups.csv")
RESULTS_DIR = os.path.join(HERE, "Results")

BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 0.0005
FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patient_of_each_image(dataset):
    with open(GROUPS_FILE) as handle:
        patient_of = {row["path"]: row["patient"] for row in csv.DictReader(handle)}
    return [patient_of[os.path.relpath(path, DATASET).replace(os.sep, "/")] for path, _ in dataset.samples]


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def predict(model, loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in loader:
            predictions.extend(model(inputs.to(device)).argmax(1).cpu().tolist())
    return predictions


def train_and_test(train_set, test_set, train_idx, test_idx, num_classes, seed):
    set_seed(seed)
    model = build_model(num_classes)
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(Subset(train_set, train_idx), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(Subset(test_set, test_idx), batch_size=BATCH_SIZE, shuffle=False)
    true_labels = [test_set.targets[i] for i in test_idx]

    correct_by_epoch = []
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        predictions = predict(model, test_loader)
        correct_by_epoch.append(sum(p == t for p, t in zip(predictions, true_labels)))
    return predictions, correct_by_epoch


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_set = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    test_set = datasets.ImageFolder(DATA_DIR, transform=test_transforms)
    classes = train_set.classes
    labels = np.array(train_set.targets)
    patients = patient_of_each_image(train_set)
    print(f"{len(labels)} images from {len(set(patients))} patients, classes {classes}")

    runs = []
    summed_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for seed in SEEDS:
        folds = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        final_correct = 0
        picked_correct = 0
        true_all, predicted_all = [], []
        for fold, (train_idx, test_idx) in enumerate(folds.split(np.zeros(len(labels)), labels, patients)):
            assert not set(patients[i] for i in train_idx) & set(patients[i] for i in test_idx)
            predictions, correct_by_epoch = train_and_test(train_set, test_set, train_idx.tolist(), test_idx.tolist(), len(classes), seed * 100 + fold)
            final_correct += correct_by_epoch[-1]
            picked_correct += max(correct_by_epoch)
            true_all.extend(labels[test_idx].tolist())
            predicted_all.extend(predictions)
        matrix = confusion_matrix(true_all, predicted_all, labels=list(range(len(classes))))
        summed_matrix += matrix
        runs.append({
            "seed": seed,
            "accuracy_after_40_epochs": final_correct / len(labels),
            "accuracy_if_best_epoch_is_picked_on_the_test_fold": picked_correct / len(labels),
            "confusion_matrix": matrix.tolist(),
        })
        print(f"seed {seed}: {final_correct}/{len(labels)} = {final_correct / len(labels):.1%} after 40 epochs | {picked_correct / len(labels):.1%} if the best epoch is picked on the test fold")

    final = np.array([run["accuracy_after_40_epochs"] for run in runs])
    picked = np.array([run["accuracy_if_best_epoch_is_picked_on_the_test_fold"] for run in runs])
    counts = np.bincount(labels)
    recall = summed_matrix.diagonal() / summed_matrix.sum(axis=1)
    print(f"\npatient-grouped {FOLDS}-fold accuracy: mean {final.mean():.1%} over {len(SEEDS)} seeds (lowest {final.min():.1%}, highest {final.max():.1%})")
    print(f"picking the best epoch on the test fold, the way the old 76.92% was made: mean {picked.mean():.1%}")
    print(f"always answering {classes[counts.argmax()]}: {counts.max() / len(labels):.1%}")
    print(f"confusion matrix summed over seeds, rows = true {classes}, columns = predicted")
    print(summed_matrix)
    for name, value in zip(classes, recall):
        print(f"   {name}: {value:.1%} of its images recognised")

    with open(os.path.join(RESULTS_DIR, "Step2_Old_Classifier_CV.json"), "w") as handle:
        json.dump({
            "classes": classes,
            "images": len(labels),
            "patients": len(set(patients)),
            "folds": FOLDS,
            "mean_accuracy": final.mean(),
            "lowest_accuracy": final.min(),
            "highest_accuracy": final.max(),
            "mean_accuracy_with_best_epoch_picked_on_test_fold": picked.mean(),
            "majority_class_accuracy": counts.max() / len(labels),
            "summed_confusion_matrix": summed_matrix.tolist(),
            "runs": runs,
        }, handle, indent=1)


if __name__ == "__main__":
    main()
