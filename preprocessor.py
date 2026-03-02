import os
import re

import torch
from torch.utils.data import Dataset

from torchvision import transforms

from PIL import Image
from collections import Counter 

from config import CLASS_NAMES




# Transforms
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
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



# Dataset (image + text_vec)
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
                if f.lower().endswith((".jpg",".jpeg",".png")):
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
    
# Quick count
def count_images(root_dir):
    total = 0
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        total += len([f for f in os.listdir(cls_dir)
                      if f.lower().endswith((".jpg",".jpeg",".png"))])
    return total




def build_vocab_from_dirs(dirs, class_names, max_vocab=5000, min_freq=2):
    counter = Counter()
    for root in dirs:
        for clas in class_names:
            cls_dir = os.path.join(root, clas)
            if not os.path.exists(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".jpg",".jpeg",".png")):
                    counter.update(tokenize(filename_to_text(f)))

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.most_common():
        if freq >= min_freq and len(vocab) < max_vocab:
            vocab[word] = len(vocab)
    return vocab

# Text processing + vocab
def filename_to_text(fname):
    base = os.path.splitext(fname)[0]
    base = re.sub(r"_\d+$", "", base)
    return base.replace("_", " ").strip()

def tokenize(text):
    return re.findall(r"[a-zA-Z]+", text.lower())