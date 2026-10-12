import os
import time
import random
import datetime
import itertools

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from Networks import UNetGenerator, CycleGANDiscriminator, weights_init_normal

HERE = os.path.dirname(os.path.abspath(__file__))
BE_PROJECT = os.path.abspath(os.path.join(HERE, ".."))
DATASET = os.path.join(BE_PROJECT, "Dataset")
KEPT_FILES = os.path.join(DATASET, "Splits", "cyclegan_kept_files.txt")
WEIGHTS_DIR = os.path.join(BE_PROJECT, "cycleGAN_Weights_TrainOnly")
RESUME_FILE = os.path.join(WEIGHTS_DIR, "resume_checkpoint.pth")

SEED = 2026
IMG_SIZE = 256
LOAD_SIZE = 286
BATCH_SIZE = 1
NUM_WORKERS = 2
LAST_EPOCH = 95
LR = 0.0002
B1 = 0.5
B2 = 0.999
LAMBDA_CYC = 5.0
LAMBDA_ID = 0.0
PRINT_EVERY = 50


def big_enough(path):
    with Image.open(path) as image:
        width, height = image.size
    return width >= LOAD_SIZE and height >= LOAD_SIZE


def read_kept_files():
    with open(KEPT_FILES) as handle:
        paths = [os.path.join(DATASET, line.strip()) for line in handle if line.strip()]
    cf_paths = [path for path in paths if os.path.basename(path).startswith("CF_")]
    faf_paths = [path for path in paths if os.path.basename(path).startswith("FAF_")]
    return cf_paths, faf_paths


class UnpairedFundusDataset(Dataset):
    def __init__(self, cf_paths, faf_paths, transform):
        self.cf_images = [path for path in cf_paths if big_enough(path)]
        self.faf_images = [path for path in faf_paths if big_enough(path)]
        self.transform = transform
        print(f"[TRAIN] CF Images: {len(self.cf_images)} | FAF Images: {len(self.faf_images)}")

    def __len__(self):
        return max(len(self.cf_images), len(self.faf_images))

    def __getitem__(self, index):
        cf_path = self.cf_images[index % len(self.cf_images)]
        faf_path = self.faf_images[random.randint(0, len(self.faf_images) - 1)]
        cf_image = Image.open(cf_path).convert("RGB")
        faf_image = Image.open(faf_path).convert("L").convert("RGB")
        return {"A": self.transform(cf_image), "B": self.transform(faf_image)}


class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            elif random.uniform(0, 1) > 0.5:
                i = random.randint(0, self.max_size - 1)
                to_return.append(self.data[i].clone())
                self.data[i] = element
            else:
                to_return.append(element)
        return torch.cat(to_return)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((LOAD_SIZE, LOAD_SIZE), Image.BICUBIC),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    cf_paths, faf_paths = read_kept_files()
    dataset = UnpairedFundusDataset(cf_paths, faf_paths, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)

    G_AB = UNetGenerator(in_channels=3, features=48).to(device)
    G_BA = UNetGenerator(in_channels=3, features=48).to(device)
    D_A = CycleGANDiscriminator().to(device)
    D_B = CycleGANDiscriminator().to(device)
    for model in (G_AB, G_BA, D_A, D_B):
        model.apply(weights_init_normal)

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=LR, betas=(B1, B2))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=LR, betas=(B1, B2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=LR, betas=(B1, B2))
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    parts = {"G_AB": G_AB, "G_BA": G_BA, "D_A": D_A, "D_B": D_B,
             "optimizer_G": optimizer_G, "optimizer_D_A": optimizer_D_A, "optimizer_D_B": optimizer_D_B}
    start_epoch = 0
    if os.path.exists(RESUME_FILE):
        state = torch.load(RESUME_FILE, map_location=device, weights_only=False)
        for name, part in parts.items():
            part.load_state_dict(state[name])
        start_epoch = state["epoch"] + 1
        print(f"Resuming after epoch {state['epoch']}")

    patch = (1, IMG_SIZE // 16, IMG_SIZE // 16)
    total_batches = (LAST_EPOCH + 1 - start_epoch) * len(dataloader)
    batches_done = 0
    start_time = time.time()
    print(f"--- Starting Training on {device} ---", flush=True)

    for epoch in range(start_epoch, LAST_EPOCH + 1):
        for i, batch in enumerate(dataloader):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)
            valid = torch.ones(real_A.size(0), *patch, device=device)
            fake = torch.zeros(real_A.size(0), *patch, device=device)

            G_AB.train()
            G_BA.train()
            optimizer_G.zero_grad()
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            rec_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(rec_A, real_A)
            rec_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(rec_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            loss_G = loss_GAN + LAMBDA_CYC * loss_cycle + LAMBDA_ID * loss_identity
            loss_G.backward()
            optimizer_G.step()

            optimizer_D_A.zero_grad()
            loss_real = criterion_GAN(D_A(real_A), valid)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), valid)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            batches_done += 1
            if i % PRINT_EVERY == 0:
                seconds_per_batch = (time.time() - start_time) / batches_done
                time_left = datetime.timedelta(seconds=int((total_batches - batches_done) * seconds_per_batch))
                print(f"[Epoch {epoch}/{LAST_EPOCH}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {(loss_D_A + loss_D_B).item():.4f}] [G loss: {loss_G.item():.4f}] "
                      f"[{seconds_per_batch:.3f} s/batch] ETA: {time_left}", flush=True)

        if epoch % 5 == 0 and epoch != LAST_EPOCH:
            torch.save({"epoch": epoch, **{name: part.state_dict() for name, part in parts.items()}}, RESUME_FILE)
            print(f"Saved resume checkpoint after epoch {epoch}", flush=True)

    torch.save(G_AB.state_dict(), os.path.join(WEIGHTS_DIR, f"G_AB_{LAST_EPOCH}.pth"))
    torch.save(G_BA.state_dict(), os.path.join(WEIGHTS_DIR, f"G_BA_{LAST_EPOCH}.pth"))
    if os.path.exists(RESUME_FILE):
        os.remove(RESUME_FILE)
    print(f"Training finished in {datetime.timedelta(seconds=int(time.time() - start_time))}. Saved G_AB_{LAST_EPOCH}.pth and G_BA_{LAST_EPOCH}.pth", flush=True)


if __name__ == "__main__":
    main()
