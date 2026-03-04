import os
import re
import numpy as np
from collections import Counter 

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from custom_augmentation import (
    RandomColorTemperature, AddGaussianNoise, RandomDownsample,
    RandomCutout, GarbageSpecificAugmentation
)
from config import CLASS_NAMES


# ==================== TRANSFORMS ====================
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  
        transforms.RandomRotation(20),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                   saturation=0.3, hue=0.1)
        ], p=0.6),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        AddGaussianNoise(std_range=(0.01, 0.05), p=0.5),
        RandomDownsample(scale_range=(0.6, 0.95), p=0.3),
        RandomColorTemperature(strength=0.15, p=0.4),
        RandomCutout(scale_range=(0.02, 0.15), p=0.3),
        GarbageSpecificAugmentation(p=0.2),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
}


# ==================== DATASET ====================
class ImageTextGarbageDataset(Dataset):
    def __init__(self, root_dir, transform=None, vocab=None, class_names=None):
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = vocab
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        self.samples = []
        for cls in class_names:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((
                        os.path.join(cls_dir, f),
                        filename_to_text(f),
                        self.class_to_idx[cls]
                    ))

    def __len__(self):
        return len(self.samples)

    def encode_text_bow(self, text):
        vec = torch.zeros(len(self.vocab), dtype=torch.float32)
        for w in tokenize(text):
            vec[self.vocab.get(w, self.vocab["<unk>"])] += 1.0
        if vec.sum() > 0:
            vec /= vec.sum()
        return vec

    def __getitem__(self, idx):
        path, text, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "text_vec": self.encode_text_bow(text),
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
            "text": text
        }


# ==================== UTILITY FUNCTIONS ====================
def count_images(root_dir):
    total = 0
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        total += len([f for f in os.listdir(cls_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    return total


def build_vocab_from_dirs(dirs, class_names, max_vocab=5000, min_freq=2):
    counter = Counter()
    for root in dirs:
        for clas in class_names:
            cls_dir = os.path.join(root, clas)
            if not os.path.exists(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    counter.update(tokenize(filename_to_text(f)))

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.most_common():
        if freq >= min_freq and len(vocab) < max_vocab:
            vocab[word] = len(vocab)
    return vocab


def compute_class_weights(dataset):
    """
    Compute weights inversely proportional to class frequencies.
    Handles unbalanced data by giving more weight to minority classes.
    """
    labels = [sample[2] for sample in dataset.samples]
    counts = Counter(labels)
    max_count = max(counts.values())
    weights = [max_count / counts[i] for i in range(len(dataset.class_names))]
    return torch.tensor(weights, dtype=torch.float32)


def filename_to_text(fname):
    base = os.path.splitext(fname)[0]
    base = re.sub(r"_\d+$", "", base)
    return base.replace("_", " ").strip()


def tokenize(text):
    return re.findall(r"[a-zA-Z]+", text.lower())