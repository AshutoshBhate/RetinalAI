import os
import csv
import json
import random
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from PIL import Image
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
BE_PROJECT = os.path.abspath(os.path.join(HERE, ".."))
DATASET = os.path.join(BE_PROJECT, "Dataset")
SPLITS = os.path.join(DATASET, "Splits")
RESULTS_DIR = os.path.join(HERE, "Results")
WEIGHTS_DIR = os.path.join(BE_PROJECT, "Codes", "Training", "Training_3_classes_TrainOnly")

SOURCES = {
    "Generated FAF (CycleGAN without test patients)": os.path.join(DATASET, "Generated_FAF_CNN_Dataset_3classes_TrainOnlyGAN"),
    "Real CF photos, no GAN": os.path.join(DATASET, "Final_Merged_CF_Renumbered"),
}
CLASSES = ["Choroiditis", "Normal", "Scar"]

BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 0.0005
FOLDS = 5
CV_SEEDS = [0, 1, 2, 3, 4]
FINAL_SEED = 2026

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


def read_lines(name):
    with open(os.path.join(SPLITS, name)) as handle:
        return [line.strip() for line in handle if line.strip()]


@lru_cache(maxsize=None)
def load_at_gan_size(path):
    return Image.open(path).convert("RGB").resize((256, 256), Image.BICUBIC)


class ClassifierImages(datasets.ImageFolder):
    def find_classes(self, directory):
        return CLASSES, {name: index for index, name in enumerate(CLASSES)}


def source_cf(path):
    class_name = os.path.basename(os.path.dirname(path))
    name = os.path.basename(path).replace("GenFAF_", "CF_")
    return f"Final_Merged_CF_Renumbered/{class_name}/{name}"


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(device)


def train(train_set, indices, seed):
    set_seed(seed)
    model = build_model()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(Subset(train_set, indices), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed))
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
    return model


def predict(model, test_set, indices):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in DataLoader(Subset(test_set, indices), batch_size=BATCH_SIZE, shuffle=False):
            predictions.extend(model(inputs.to(device)).argmax(1).cpu().tolist())
    return predictions


def evaluate_source(name, folder, patient_of, train_cfs, test_cfs):
    wanted = train_cfs | test_cfs
    train_set = ClassifierImages(folder, transform=train_transforms, loader=load_at_gan_size, is_valid_file=lambda path: source_cf(path) in wanted)
    test_set = ClassifierImages(folder, transform=test_transforms, loader=load_at_gan_size, is_valid_file=lambda path: source_cf(path) in wanted)
    labels = np.array(train_set.targets)
    sources = [source_cf(path) for path, _ in train_set.samples]
    train_idx = [i for i, source in enumerate(sources) if source in train_cfs]
    test_idx = [i for i, source in enumerate(sources) if source in test_cfs]
    patients = np.array([patient_of[source] for source in sources])
    print(f"\n== {name}: {len(train_idx)} training images ({len(set(patients[train_idx]))} patients), {len(test_idx)} test images ({len(set(patients[test_idx]))} patients)")

    cv_accuracies = []
    for seed in CV_SEEDS:
        folds = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        correct = 0
        for fold, (inner_train, inner_test) in enumerate(folds.split(np.zeros(len(train_idx)), labels[train_idx], patients[train_idx])):
            fold_train = [train_idx[i] for i in inner_train]
            fold_test = [train_idx[i] for i in inner_test]
            model = train(train_set, fold_train, seed * 100 + fold)
            correct += sum(p == labels[i] for p, i in zip(predict(model, test_set, fold_test), fold_test))
        cv_accuracies.append(correct / len(train_idx))
    print(f"patient-grouped {FOLDS}-fold accuracy on the training patients: mean {np.mean(cv_accuracies):.1%} over {len(CV_SEEDS)} seeds "
          f"(lowest {min(cv_accuracies):.1%}, highest {max(cv_accuracies):.1%})")

    model = train(train_set, train_idx, FINAL_SEED)
    predictions = predict(model, test_set, test_idx)
    true = labels[test_idx].tolist()
    matrix = confusion_matrix(true, predictions, labels=list(range(len(CLASSES))))
    correct = int(matrix.trace())
    print(f"one-time test on the held-out patients: {correct}/{len(test_idx)} = {correct / len(test_idx):.1%}")
    print(f"confusion matrix, rows = true {CLASSES}, columns = predicted")
    print(matrix)
    return model, {
        "training_images": len(train_idx),
        "test_images": len(test_idx),
        "cv_mean_accuracy": float(np.mean(cv_accuracies)),
        "cv_lowest_accuracy": float(min(cv_accuracies)),
        "cv_highest_accuracy": float(max(cv_accuracies)),
        "test_correct": correct,
        "test_accuracy": correct / len(test_idx),
        "test_confusion_matrix": matrix.tolist(),
        "test_predictions": {sources[i]: CLASSES[p] for i, p in zip(test_idx, predictions)},
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    with open(os.path.join(SPLITS, "patient_groups.csv")) as handle:
        patient_of = {row["path"]: row["patient"] for row in csv.DictReader(handle)}
    train_cfs = set(read_lines("classifier_cf_train.txt"))
    test_cfs = set(read_lines("classifier_cf_test.txt"))
    test_classes = [path.split("/")[1] for path in sorted(test_cfs)]
    majority = max(test_classes.count(name) for name in CLASSES) / len(test_classes)
    print(f"always answering the biggest class on the test images: {majority:.1%}")

    results = {"majority_class_test_accuracy": majority}
    for name, folder in SOURCES.items():
        model, results[name] = evaluate_source(name, folder, patient_of, train_cfs, test_cfs)
        if name.startswith("Generated FAF"):
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "resnet_faf_classifier.pth"))
            print(f"saved {os.path.join(WEIGHTS_DIR, 'resnet_faf_classifier.pth')}")

    with open(os.path.join(RESULTS_DIR, "Step7_Classifier_TrainOnly.json"), "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
