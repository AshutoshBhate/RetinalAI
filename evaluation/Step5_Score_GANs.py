import os
import csv
import json
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from Networks import UNetGenerator

HERE = os.path.dirname(os.path.abspath(__file__))
BE_PROJECT = os.path.abspath(os.path.join(HERE, ".."))
DATASET = os.path.join(BE_PROJECT, "Dataset")
SPLITS = os.path.join(DATASET, "Splits")
RESULTS_DIR = os.path.join(HERE, "Results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2026
BOOTSTRAP_ROUNDS = 2000

GENERATORS = {
    "Old CycleGAN (saw every patient)": os.path.join(BE_PROJECT, "cycleGAN_Weights", "G_AB_95.pth"),
    "CycleGAN without test patients": os.path.join(BE_PROJECT, "cycleGAN_Weights_TrainOnly", "G_AB_95.pth"),
    "Pix2Pix without test patients": os.path.join(BE_PROJECT, "pix2pix_Weights_TrainOnly", "best_gen.pth"),
}

preprocess = transforms.Compose([
    transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def read_lines(name):
    with open(os.path.join(SPLITS, name)) as handle:
        return [line.strip() for line in handle if line.strip()]


def faf_partner(cf_path):
    folder, name = os.path.split(cf_path)
    return os.path.join(folder, "FAF_" + name[len("CF_"):])


def load_rgb(relative):
    return Image.open(os.path.join(DATASET, relative)).convert("RGB")


def as_array(image):
    return np.asarray(image.resize((256, 256), Image.BICUBIC), dtype=np.float32) / 255.0


def translate(generator, image):
    with torch.no_grad():
        fake = generator(preprocess(image).unsqueeze(0).to(DEVICE))
    return (fake[0] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def scores(real, fake):
    return {
        "ssim": float(compute_ssim(real, fake, data_range=1.0, channel_axis=-1, win_size=7)),
        "psnr": float(compute_psnr(real, fake, data_range=1.0)),
        "mae": float(np.mean(np.abs(real - fake))),
    }


def score_pairs(name, predict, pairs, rows):
    results = []
    for path in pairs:
        real = as_array(load_rgb(faf_partner(path)))
        disease = rows[path]["disease"] if path in rows else path.split("/")[1]
        patient = rows[path]["patient"] if path in rows else ""
        results.append({"model": name, "cf": path, "disease": disease, "patient": patient, **scores(real, predict(load_rgb(path)))})
    return results


def patient_bootstrap(values, patients):
    by_patient = defaultdict(list)
    for value, patient in zip(values, patients):
        by_patient[patient].append(value)
    groups = list(by_patient.values())
    rng = np.random.default_rng(SEED)
    means = []
    for _ in range(BOOTSTRAP_ROUNDS):
        picked = rng.integers(0, len(groups), len(groups))
        means.append(np.mean([value for index in picked for value in groups[index]]))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(SPLITS, "patient_groups.csv")) as handle:
        rows = {row["path"]: row for row in csv.DictReader(handle)}
    test_pairs = read_lines("gan_test_pairs.txt")
    train_pairs = read_lines("pix2pix_train_pairs.txt")
    patients = [rows[path]["patient"] for path in test_pairs]
    print(f"test pairs: {len(test_pairs)} from {len(set(patients))} test patients")

    mean_train_faf = np.mean([as_array(load_rgb(faf_partner(path))) for path in train_pairs], axis=0)
    predictions = {
        "Baseline: grey copy of the CF photo": lambda image: as_array(image.convert("L").convert("RGB")),
        "Baseline: average training FAF": lambda image: mean_train_faf,
    }
    for name, weights in GENERATORS.items():
        if not os.path.exists(weights):
            print(f"skipping {name}: {weights} not found")
            continue
        generator = UNetGenerator(in_channels=3, features=48)
        generator.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        generator = generator.to(DEVICE).eval()
        predictions[name] = lambda image, generator=generator: translate(generator, image)

    per_pair = []
    summary = {}
    for name, predict in predictions.items():
        results = score_pairs(name, predict, test_pairs, rows)
        per_pair.extend(results)
        ssim_values = [result["ssim"] for result in results]
        low, high = patient_bootstrap(ssim_values, patients)
        summary[name] = {
            "ssim_mean": float(np.mean(ssim_values)),
            "ssim_median": float(np.median(ssim_values)),
            "ssim_95_range_over_patients": [low, high],
            "psnr_mean": float(np.mean([result["psnr"] for result in results])),
            "mae_mean": float(np.mean([result["mae"] for result in results])),
            "ssim_by_disease": {disease: float(np.mean([r["ssim"] for r in results if r["disease"] == disease]))
                                for disease in sorted({r["disease"] for r in results})},
        }
        print(f"{name}: SSIM {summary[name]['ssim_mean']:.4f} (median {summary[name]['ssim_median']:.4f}, "
              f"95% range {low:.3f}-{high:.3f}) | PSNR {summary[name]['psnr_mean']:.2f} dB | MAE {summary[name]['mae_mean']:.4f}")

    print("\nmean SSIM by disease:")
    for disease in sorted({row["disease"] for row in per_pair}):
        count = sum(1 for path in test_pairs if rows[path]["disease"] == disease)
        line = " | ".join(f"{name.split(' (')[0].replace('Baseline: ', '')}: {summary[name]['ssim_by_disease'][disease]:.3f}" for name in summary)
        print(f"   {disease} ({count} pairs): {line}")

    old_name = "Old CycleGAN (saw every patient)"
    if old_name in predictions:
        all_pairs = sorted(train_pairs + test_pairs)
        old_results = score_pairs(old_name + ", all master pairs", predictions[old_name], all_pairs, rows)
        old_ssim = np.array([result["ssim"] for result in old_results])
        best = max(old_results, key=lambda result: result["ssim"])
        summary[old_name + ", all master pairs"] = {
            "pairs": len(old_results),
            "ssim_mean": float(old_ssim.mean()),
            "ssim_median": float(np.median(old_ssim)),
            "ssim_best_single_pair": best["ssim"],
            "best_pair": best["cf"],
            "share_above_0.2333": float(np.mean(old_ssim > 0.2333)),
            "psnr_mean": float(np.mean([result["psnr"] for result in old_results])),
            "mae_mean": float(np.mean([result["mae"] for result in old_results])),
        }
        print(f"\n{old_name} on all {len(old_results)} master pairs (not a test, it trained on these photos): "
              f"SSIM mean {old_ssim.mean():.4f}, median {np.median(old_ssim):.4f}, best single pair {best['ssim']:.4f} ({os.path.basename(best['cf'])}), "
              f"{np.mean(old_ssim > 0.2333):.1%} of pairs above 0.2333")
        per_pair.extend(old_results)

    with open(os.path.join(RESULTS_DIR, "Step5_GAN_scores_per_pair.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "cf", "disease", "patient", "ssim", "psnr", "mae"])
        writer.writeheader()
        writer.writerows(per_pair)
    with open(os.path.join(RESULTS_DIR, "Step5_GAN_scores.json"), "w") as handle:
        json.dump({"test_pairs": len(test_pairs), "test_patients": len(set(patients)), "models": summary}, handle, indent=1)


if __name__ == "__main__":
    main()
