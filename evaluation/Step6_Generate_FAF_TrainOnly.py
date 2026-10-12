import os

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from Networks import UNetGenerator

HERE = os.path.dirname(os.path.abspath(__file__))
BE_PROJECT = os.path.abspath(os.path.join(HERE, ".."))
DATASET = os.path.join(BE_PROJECT, "Dataset")
SPLITS = os.path.join(DATASET, "Splits")
WEIGHTS = os.path.join(BE_PROJECT, "cycleGAN_Weights_TrainOnly", "G_AB_95.pth")
OUTPUT_DIR = os.path.join(DATASET, "Generated_FAF_CNN_Dataset_3classes_TrainOnlyGAN")


def read_lines(name):
    with open(os.path.join(SPLITS, name)) as handle:
        return [line.strip() for line in handle if line.strip()]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_AB = UNetGenerator(in_channels=3, features=48).to(device)
    G_AB.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
    G_AB.eval()

    img_transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    cf_paths = read_lines("classifier_cf_train.txt") + read_lines("classifier_cf_test.txt")
    for cf_path in cf_paths:
        class_name = cf_path.split("/")[1]
        target_class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(target_class_dir, exist_ok=True)
        img = Image.open(os.path.join(DATASET, cf_path)).convert("RGB")
        with torch.no_grad():
            fake_faf_tensor = G_AB(img_transform(img).unsqueeze(0).to(device))
        new_filename = os.path.basename(cf_path).replace("CF_", "GenFAF_")
        save_image(fake_faf_tensor, os.path.join(target_class_dir, new_filename), normalize=True, value_range=(-1, 1))
    print(f"{len(cf_paths)} FAF images generated in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
