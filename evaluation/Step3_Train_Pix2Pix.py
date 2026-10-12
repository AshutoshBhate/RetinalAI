import os
import csv
import json
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim

from Networks import UNetGenerator, Pix2PixDiscriminator

HERE = os.path.dirname(os.path.abspath(__file__))
BE_PROJECT = os.path.abspath(os.path.join(HERE, ".."))
DATASET = os.path.join(BE_PROJECT, "Dataset")
SPLITS = os.path.join(DATASET, "Splits")
WEIGHTS_DIR = os.path.join(BE_PROJECT, "pix2pix_Weights_TrainOnly")
RESULTS_DIR = os.path.join(HERE, "Results")

SEED = 2026
VALIDATION_FRACTION = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
IMAGE_SIZE = 256
L1_LAMBDA = 100
NUM_EPOCHS = 200
PATIENCE = 30
MIN_DELTA = 0.001


def read_lines(name):
    with open(os.path.join(SPLITS, name)) as handle:
        return [line.strip() for line in handle if line.strip()]


def faf_partner(cf_path):
    folder, name = os.path.split(cf_path)
    return os.path.join(folder, "FAF_" + name[len("CF_"):])


def split_validation(train_pairs):
    with open(os.path.join(SPLITS, "patient_groups.csv")) as handle:
        rows = {row["path"]: row for row in csv.DictReader(handle)}
    patients_by_disease = defaultdict(set)
    for path in train_pairs:
        if path in rows:
            patients_by_disease[rows[path]["disease"]].add(rows[path]["patient"])

    rng = random.Random(SEED)
    validation_patients = set()
    for disease in sorted(patients_by_disease):
        patients = sorted(patients_by_disease[disease])
        rng.shuffle(patients)
        validation_patients.update(patients[:max(1, round(VALIDATION_FRACTION * len(patients)))])

    validation = [path for path in train_pairs if path in rows and rows[path]["patient"] in validation_patients]
    training = [path for path in train_pairs if path not in validation]
    return training, validation, validation_patients


class FundusDataset(Dataset):
    def __init__(self, file_list, image_size=256, is_train=True):
        self.list_files = file_list
        self.image_size = image_size
        if is_train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=40, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                    ToTensorV2(),
                ],
                additional_targets={"image0": "image"},
            )
        else:
            self.transform = A.Compose(
                [
                    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                    ToTensorV2(),
                ],
                additional_targets={"image0": "image"},
            )

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        cf_file_path = os.path.join(DATASET, self.list_files[index])
        faf_file_path = faf_partner(cf_file_path)
        final_dim = (self.image_size, self.image_size)
        cf_image = np.array(Image.open(cf_file_path).convert("RGB").resize(final_dim, resample=Image.Resampling.BICUBIC))
        faf_image = np.array(Image.open(faf_file_path).convert("RGB").resize(final_dim, resample=Image.Resampling.BICUBIC))
        transformed = self.transform(image=cf_image, image0=faf_image)
        return transformed["image"], transformed["image0"]


def validate_epoch(gen, loader):
    gen.eval()
    ssim_scores = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_fake = gen(x)
            real_img = (y[0] * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy()
            fake_img = (y_fake[0] * 0.5 + 0.5).float().permute(1, 2, 0).cpu().numpy()
            ssim_scores.append(ssim(real_img, fake_img, data_range=1.0, channel_axis=-1, win_size=7))
    gen.train()
    return float(np.mean(ssim_scores))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    training, validation, validation_patients = split_validation(read_lines("pix2pix_train_pairs.txt"))
    with open(os.path.join(SPLITS, "pix2pix_validation_pairs.txt"), "w", newline="\n") as handle:
        handle.writelines(path + "\n" for path in validation)
    print(f"Training pairs: {len(training)} | Validation pairs: {len(validation)} from {len(validation_patients)} training patients")

    gen = UNetGenerator(in_channels=3, features=48).to(DEVICE)
    disc = Pix2PixDiscriminator(in_channels=3).to(DEVICE)
    opt_disc = optim.AdamW(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.AdamW(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    scheduler_gen = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_gen, T_0=25, T_mult=2, eta_min=1e-6)
    scheduler_disc = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_disc, T_0=25, T_mult=2, eta_min=1e-6)
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    train_loader = DataLoader(FundusDataset(training, IMAGE_SIZE, is_train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(FundusDataset(validation, IMAGE_SIZE, is_train=False), batch_size=1, shuffle=False)

    best_ssim = -1.0
    best_epoch = 0
    early_stop_best = -np.inf
    counter = 0
    history = []
    for epoch in range(NUM_EPOCHS):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE):
                y_fake = gen(x)
                D_real = disc(x, y)
                D_real_loss = BCE(D_real, torch.ones_like(D_real))
                D_fake = disc(x, y_fake.detach())
                D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2
            opt_disc.zero_grad()
            D_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_disc.step()
            with torch.amp.autocast(device_type=DEVICE):
                D_fake = disc(x, y_fake)
                G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
                L1 = L1_LOSS(y_fake, y) * L1_LAMBDA
                G_loss = G_fake_loss + L1
            opt_gen.zero_grad()
            G_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt_gen.step()
        scheduler_gen.step()
        scheduler_disc.step()

        val_ssim = validate_epoch(gen, val_loader)
        history.append(val_ssim)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | G_loss {G_loss.item():.3f} D_loss {D_loss.item():.3f} | Validation SSIM: {val_ssim:.4f} (Best SSIM: {max(best_ssim, val_ssim):.4f})", flush=True)
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            best_epoch = epoch + 1
            torch.save(gen.state_dict(), os.path.join(WEIGHTS_DIR, "best_gen.pth"))
        if val_ssim > early_stop_best + MIN_DELTA:
            early_stop_best = val_ssim
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print("Early stopping triggered! Training has finished.")
                break

    with open(os.path.join(RESULTS_DIR, "Step3_Pix2Pix_training.json"), "w") as handle:
        json.dump({
            "training_pairs": len(training),
            "validation_pairs": len(validation),
            "validation_patients": sorted(validation_patients),
            "best_epoch": best_epoch,
            "best_validation_ssim": best_ssim,
            "validation_ssim_by_epoch": history,
        }, handle, indent=1)
    print(f"Best validation SSIM {best_ssim:.4f} at epoch {best_epoch}. Saved best_gen.pth")


if __name__ == "__main__":
    main()
