import os
import csv
import random
import hashlib
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BE_PROJECT = os.path.abspath(os.path.join(HERE, ".."))
DATASET = os.path.join(BE_PROJECT, "Dataset")
SPLITS = os.path.join(DATASET, "Splits")
SIZE_LISTS = os.path.join(SPLITS, "patient_sizes")

SEED = 2026
TEST_FRACTION = 0.2
CLASSIFIER_CLASSES = ["Choroiditis", "Normal", "Scar"]
IMAGE_TYPES = (".jpg", ".jpeg", ".png")
SAME_EYE_POINTS = 15

sift = cv2.SIFT_create(nfeatures=1000)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
matcher = cv2.BFMatcher(cv2.NORM_L2)

SCANNED_FOLDERS = [
    "Actual_Original_Master",
    "Clean_CF",
    "Clean_Bhushan_CF",
    "Final_Merged_CF",
    "Final_Merged_CF_Renumbered",
    "CycleGAN_Data_Unpaired_CF-FAF",
    "Generated_FAF_CNN_Dataset_Refactored_3classes",
    "Test_CF_Large",
    "Test_CF_Small",
]

DISEASE_FOLDER_NAMES = {
    "Choroiditis": "Choroiditis",
    "SCAR": "Scar",
    "Scar": "Scar",
    "Normal": "Normal",
    "CNVM": "CNVM",
    "CSR": "CSR",
    "Atrophy": "Atrophy",
    "Stargardt": "Stargardt",
}


def read_size_lists():
    size_to_patient = {}
    patient_disease = {}
    for file_name in sorted(os.listdir(SIZE_LISTS)):
        disease = os.path.splitext(file_name)[0]
        size_to_patient[disease] = {}
        with open(os.path.join(SIZE_LISTS, file_name)) as handle:
            for line in handle:
                size, patient = line.split()
                size_to_patient[disease][int(size)] = patient
                patient_disease[patient] = disease
    return size_to_patient, patient_disease


def scan_images():
    images = []
    for folder in SCANNED_FOLDERS:
        root = os.path.join(DATASET, folder)
        for current, _, files in os.walk(root):
            for name in sorted(files):
                if not name.lower().endswith(IMAGE_TYPES):
                    continue
                path = os.path.join(current, name)
                relative = os.path.relpath(path, DATASET).replace(os.sep, "/")
                with open(path, "rb") as handle:
                    digest = hashlib.md5(handle.read()).hexdigest()
                images.append({"path": relative, "bytes": os.path.getsize(path), "md5": digest})
    return images


def assign_patients(images, size_to_patient):
    patient_of = {}

    sizes_anywhere = defaultdict(set)
    for disease, mapping in size_to_patient.items():
        for size, patient in mapping.items():
            sizes_anywhere[size].add(patient)

    for image in images:
        parts = image["path"].split("/")
        if parts[0].startswith("Generated"):
            continue
        if len(parts) >= 3 and parts[1] in DISEASE_FOLDER_NAMES:
            disease = DISEASE_FOLDER_NAMES[parts[1]]
            patient = size_to_patient[disease].get(image["bytes"])
            if patient:
                patient_of[image["path"]] = patient
        elif parts[0] == "CycleGAN_Data_Unpaired_CF-FAF":
            owners = sizes_anywhere.get(image["bytes"], set())
            if len(owners) == 1:
                patient_of[image["path"]] = next(iter(owners))

    paths_by_md5 = defaultdict(list)
    for image in images:
        paths_by_md5[image["md5"]].append(image["path"])

    for _ in range(3):
        for paths in paths_by_md5.values():
            owners = {patient_of[path] for path in paths if path in patient_of}
            if len(owners) > 1:
                raise ValueError(f"one image belongs to two patients: {sorted(owners)}")
            if len(owners) == 1:
                owner = owners.pop()
                for path in paths:
                    patient_of.setdefault(path, owner)

        for image in images:
            parts = image["path"].split("/")
            if parts[0] != "Actual_Original_Master" or image["path"] not in patient_of:
                continue
            name = parts[-1]
            if name.startswith("CF_"):
                partner_name = "FAF_" + name[3:]
            else:
                partner_name = "CF_" + name[4:]
            partner = "/".join(parts[:-1] + [partner_name])
            patient_of.setdefault(partner, patient_of[image["path"]])

    for image in images:
        parts = image["path"].split("/")
        if parts[0] == "Generated_FAF_CNN_Dataset_Refactored_3classes":
            source = "/".join(["Final_Merged_CF_Renumbered", parts[1], parts[2].replace("GenFAF_", "CF_")])
            if source in patient_of:
                patient_of[image["path"]] = patient_of[source]

    return patient_of


def vessel_features(relative):
    image = cv2.imdecode(np.fromfile(os.path.join(DATASET, relative), np.uint8), cv2.IMREAD_COLOR)
    if os.path.basename(relative).startswith("CF"):
        channel = image[:, :, 1]
    else:
        channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 640 / max(channel.shape)
    channel = cv2.resize(channel, (round(channel.shape[1] * scale), round(channel.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    keypoints, descriptors = sift.detectAndCompute(clahe.apply(channel), None)
    return np.float32([point.pt for point in keypoints]), descriptors


def shared_vessel_points(first, second):
    pairs = matcher.knnMatch(first[1], second[1], k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < 6:
        return 0
    source = first[0][[match.queryIdx for match in good]]
    target = second[0][[match.trainIdx for match in good]]
    transform, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if transform is None:
        return 0
    scale = np.hypot(transform[0, 0], transform[0, 1])
    return int(inliers.sum()) if 0.5 < scale < 2.0 else 0


def match_by_vessels(images, patient_of):
    cv2.setRNGSeed(SEED)
    references = {}
    for image in images:
        if image["path"] in patient_of and not image["path"].startswith(("CycleGAN", "Generated")):
            references.setdefault(image["md5"], image["path"])
    reference_features = [(path, vessel_features(path)) for path in sorted(references.values())]

    for image in images:
        path = image["path"]
        if not path.startswith("CycleGAN_Data_Unpaired_CF-FAF/") or path in patient_of:
            continue
        features = vessel_features(path)
        points, match = max((shared_vessel_points(features, other), other_path) for other_path, other in reference_features)
        if points >= SAME_EYE_POINTS:
            patient_of[path] = patient_of[match]
            print(f"{path} shows the same eye as {match} ({points} shared vessel points)")


def choose_test_patients(patient_of, patient_disease):
    present = defaultdict(set)
    for patient in set(patient_of.values()):
        present[patient_disease[patient]].add(patient)

    rng = random.Random(SEED)
    test_patients = []
    for disease in sorted(present):
        patients = sorted(present[disease])
        rng.shuffle(patients)
        n_test = max(1, round(TEST_FRACTION * len(patients)))
        if disease in CLASSIFIER_CLASSES:
            n_test = max(2, n_test)
        chosen = sorted(patients[:n_test])
        test_patients.extend(chosen)
        print(f"{disease}: {len(patients)} patients, {n_test} held out for testing")
    return set(test_patients)


def write_list(name, lines):
    with open(os.path.join(SPLITS, name), "w", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")
    print(f"wrote {name} ({len(lines)} lines)")


def main():
    size_to_patient, patient_disease = read_size_lists()
    images = scan_images()
    patient_of = assign_patients(images, size_to_patient)
    match_by_vessels(images, patient_of)
    test_patients = choose_test_patients(patient_of, patient_disease)

    with open(os.path.join(SPLITS, "patient_groups.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "disease", "patient", "role"])
        for path in sorted(patient_of):
            patient = patient_of[path]
            role = "test" if patient in test_patients else "train"
            writer.writerow([path, patient_disease[patient], patient, role])
    print(f"wrote patient_groups.csv ({len(patient_of)} images with a known patient)")

    write_list("test_patients.txt", sorted(test_patients))

    classifier_sources = []
    for cls in CLASSIFIER_CLASSES:
        for image in images:
            parts = image["path"].split("/")
            if parts[0] == "Generated_FAF_CNN_Dataset_Refactored_3classes" and parts[1] == cls:
                classifier_sources.append("/".join(["Final_Merged_CF_Renumbered", cls, parts[2].replace("GenFAF_", "CF_")]))
    classifier_sources = sorted(classifier_sources)
    write_list("classifier_cf_test.txt", [p for p in classifier_sources if patient_of[p] in test_patients])
    write_list("classifier_cf_train.txt", [p for p in classifier_sources if patient_of[p] not in test_patients])

    master_cf = sorted(i["path"] for i in images if i["path"].startswith("Actual_Original_Master/") and i["path"].split("/")[-1].startswith("CF_"))
    write_list("gan_test_pairs.txt", [p for p in master_cf if patient_of.get(p) in test_patients])
    write_list("pix2pix_train_pairs.txt", [p for p in master_cf if patient_of.get(p) not in test_patients])

    gan_files = sorted(i["path"] for i in images if i["path"].startswith("CycleGAN_Data_Unpaired_CF-FAF/"))
    write_list("cyclegan_removed_files.txt", [p for p in gan_files if patient_of.get(p) in test_patients])
    write_list("cyclegan_kept_files.txt", [p for p in gan_files if patient_of.get(p) not in test_patients])


if __name__ == "__main__":
    main()
