import os
import csv
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from Networks import UNetGenerator

HERE = os.path.dirname(os.path.abspath(__file__))
BE_PROJECT = os.path.abspath(os.path.join(HERE, ".."))
DATASET = os.path.join(BE_PROJECT, "Dataset")
RESULTS_DIR = os.path.join(HERE, "Results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXAMPLES = 3

GENERATORS = {
    "CycleGAN": os.path.join(BE_PROJECT, "cycleGAN_Weights_TrainOnly", "G_AB_95.pth"),
    "Pix2Pix": os.path.join(BE_PROJECT, "pix2pix_Weights_TrainOnly", "best_gen.pth"),
}
SCORED_MODEL = "CycleGAN without test patients"
CLASSIFIER_SOURCE = "Generated FAF (CycleGAN without test patients)"
CLASSES = ["Choroiditis", "Normal", "Scar"]

preprocess = transforms.Compose([
    transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def faf_partner(cf_path):
    folder, name = os.path.split(cf_path)
    return os.path.join(folder, "FAF_" + name[len("CF_"):])


def load_generator(weights):
    generator = UNetGenerator(in_channels=3, features=48)
    generator.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return generator.to(DEVICE).eval()


def translate(generator, image):
    with torch.no_grad():
        fake = generator(preprocess(image).unsqueeze(0).to(DEVICE))
    return (fake[0] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def median_examples():
    with open(os.path.join(RESULTS_DIR, "Step5_GAN_scores_per_pair.csv")) as handle:
        rows = [row for row in csv.DictReader(handle) if row["model"] == SCORED_MODEL]
    median = np.median([float(row["ssim"]) for row in rows])
    rows.sort(key=lambda row: abs(float(row["ssim"]) - median))
    chosen = []
    for row in rows:
        if row["disease"] not in {picked["disease"] for picked in chosen}:
            chosen.append(row)
        if len(chosen) == EXAMPLES:
            break
    return chosen, median


def held_out_examples():
    generators = {name: load_generator(path) for name, path in GENERATORS.items()}
    chosen, median = median_examples()
    columns = ["Colour fundus (input)", "Real FAF"] + [f"{name} FAF" for name in generators]
    figure, axes = plt.subplots(len(chosen), len(columns), figsize=(3.2 * len(columns), 3.4 * len(chosen)))
    for row_index, row in enumerate(chosen):
        cf_image = Image.open(os.path.join(DATASET, row["cf"])).convert("RGB")
        faf_image = Image.open(os.path.join(DATASET, faf_partner(row["cf"]))).convert("RGB")
        panels = [cf_image.resize((256, 256), Image.BICUBIC), faf_image.resize((256, 256), Image.BICUBIC)]
        panels += [translate(generator, cf_image) for generator in generators.values()]
        for column_index, panel in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(panel)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(columns[column_index], fontsize=11)
        axes[row_index, 0].set_ylabel(f"{row['disease']}\nCycleGAN SSIM {float(row['ssim']):.2f}", fontsize=10)
    figure.suptitle(f"Held-out patients: the {len(chosen)} pairs closest to the CycleGAN's median SSIM ({median:.2f}), one per disease", fontsize=11)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES_DIR, "heldout_examples.png"), dpi=100)
    plt.close(figure)


def test_confusion_matrix():
    with open(os.path.join(RESULTS_DIR, "Step7_Classifier_TrainOnly.json")) as handle:
        result = json.load(handle)[CLASSIFIER_SOURCE]
    matrix = np.array(result["test_confusion_matrix"])
    figure, axis = plt.subplots(figsize=(4.6, 4.2))
    axis.imshow(matrix, cmap="Blues")
    for true_index in range(len(CLASSES)):
        for predicted_index in range(len(CLASSES)):
            count = matrix[true_index, predicted_index]
            color = "white" if count > matrix.max() / 2 else "black"
            axis.text(predicted_index, true_index, str(count), ha="center", va="center", color=color, fontsize=13)
    axis.set_xticks(range(len(CLASSES)), CLASSES)
    axis.set_yticks(range(len(CLASSES)), CLASSES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"Held-out test: {result['test_correct']}/{result['test_images']} correct", fontsize=11)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES_DIR, "test_confusion_matrix.png"), dpi=130)
    plt.close(figure)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    held_out_examples()
    test_confusion_matrix()
    print(f"figures saved in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
