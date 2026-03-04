import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, classification_report, f1_score

from config import (
    OUT_DIR, CLASS_NAMES, TRAIN_DIR, VAL_DIR, TEST_DIR,
    set_seed, BATCH_SIZE, LABEL_SMOOTHING, EARLY_STOPPING_PATIENCE,
    WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE, PROGRESSIVE_STAGES
)

from preprocessor import (
    transform, count_images, build_vocab_from_dirs, 
    ImageTextGarbageDataset, compute_class_weights
)

from model import EfficientNetV2MMultimodalClassifier

import wandb
wandb.login(key="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")


# ==================== SETUP ====================
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("CLASS_NAMES:", CLASS_NAMES)

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    print(d, "exists:", os.path.exists(d))

print("TRAIN:", count_images(TRAIN_DIR))
print("VAL  :", count_images(VAL_DIR))
print("TEST :", count_images(TEST_DIR))

VOCAB = build_vocab_from_dirs([TRAIN_DIR, VAL_DIR], CLASS_NAMES)
VOCAB_SIZE = len(VOCAB)
print("Vocab size:", VOCAB_SIZE)

wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    mode=WANDB_MODE,
    config={
        "architecture": "EfficientNetV2-M",
        "batch_size": BATCH_SIZE,
        "label_smoothing": LABEL_SMOOTHING,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "progressive_stages": len(PROGRESSIVE_STAGES)
    }
)

# ==================== DATASETS & DATALOADERS ====================
datasets = {
    "train": ImageTextGarbageDataset(TRAIN_DIR, transform["train"], VOCAB, CLASS_NAMES),
    "val":   ImageTextGarbageDataset(VAL_DIR,   transform["val"],   VOCAB, CLASS_NAMES),
    "test":  ImageTextGarbageDataset(TEST_DIR,  transform["test"],  VOCAB, CLASS_NAMES),
}
print("Dataset sizes:", {k: len(v) for k, v in datasets.items()})

pin = device.type == "cuda"
dataloaders = {
    "train": DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=pin),
    "val":   DataLoader(datasets["val"],   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin),
    "test":  DataLoader(datasets["test"],  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin),
}

# ==================== CLASS WEIGHTS FOR UNBALANCED DATA ====================
class_weights = compute_class_weights(datasets["train"])
print("Class weights (for unbalanced data):", class_weights)

# ==================== UTILITY FUNCTIONS ====================
def set_requires_grad(module: nn.Module, flag: bool):
    # Set requires_grad for all parameters in a module.
    for p in module.parameters():
        p.requires_grad = flag


def freeze_batchnorm_running_stats(module: nn.Module):
    # Freeze BN running stats during fine-tuning with small batches.
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def count_trainable_params(model):
    # Count trainable parameters in the model.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_trainable_layers(model):
    # Print which layers are trainable.
    print("\nTrainable layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  ✓ {name}")


def get_backbone_blocks(model):
    # Get backbone blocks as a list for easier indexing.
    return list(model.image_features.children())


def make_optimizer(model: EfficientNetV2MMultimodalClassifier,
                   lr: float = 1e-4,
                   weight_decay: float = 0.01):
    # Create optimizer for trainable parameters.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


def train_model(model, loaders, criterion, optimizer, epochs, device, save_path,
                freeze_bn_stats: bool = False, stage_name: str = "", 
                best_acc_so_far: float = 0.0, stage_idx: int = 0):
    
    # Training with early stopping and best checkpoint tracking.
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    patience = 0
    best_epoch = 0

    for ep in range(epochs):
        print(f"\n{stage_name} Epoch {ep+1}/{epochs}")
        
        epoch_metrics = {"epoch": ep, "stage": stage_idx}
        
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            if phase == "train" and freeze_bn_stats:
                freeze_batchnorm_running_stats(model.image_features)

            loss_sum = 0.0
            correct = 0
            all_preds, all_labels = [], []

            for batch in tqdm(loaders[phase], leave=False):
                imgs = batch["image"].to(device)
                txts = batch["text_vec"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(imgs, txts)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)

                    if phase == "train":
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                loss_sum += loss.item() * imgs.size(0)
                correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = loss_sum / len(loaders[phase].dataset)
            epoch_acc = correct / len(loaders[phase].dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

            epoch_metrics[f"{phase}/loss"] = epoch_loss
            epoch_metrics[f"{phase}/acc"] = epoch_acc
            if phase == "val":
                epoch_metrics["val/f1"] = epoch_f1

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)
            if phase == "val":
                history["val_f1"].append(epoch_f1)

            print(f"{phase}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}, f1={epoch_f1:.4f}")

            # Save best checkpoint & early stopping for validation
            if phase == "val":
                if epoch_acc > best_acc_so_far:
                    best_acc_so_far = epoch_acc
                    best_epoch = ep
                    patience = 0
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved best model: {save_path} (val acc={best_acc_so_far:.4f})")
                else:
                    patience += 1
                    if patience >= EARLY_STOPPING_PATIENCE:
                        print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement")
                        return history, best_acc_so_far

    wandb.log(epoch_metrics)
    return history, best_acc_so_far


def save_stage_curves(history, out_dir, stage_name):
    """Save training curves for a stage."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train_loss", marker='o')
    plt.plot(history["val_loss"], label="val_loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{stage_name}: Loss vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="train_acc", marker='o')
    plt.plot(history["val_acc"], label="val_acc", marker='s')
    if "val_f1" in history:
        plt.plot(history["val_f1"], label="val_f1", marker='^')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{stage_name}: Accuracy & F1 vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{stage_name}_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()

# Find the last completed stage by checking which checkpoint files exist.
def find_last_completed_stage(out_dir):

    # Returns: (stage_num, checkpoint_path) or (-1, None) if none exist
    stage_num = -1
    last_ckpt = None
    
    # Check stages in order
    stage_idx = 0
    while True:
        ckpt_path = os.path.join(out_dir, f"stage_{stage_idx}_classifier_only.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(out_dir, f"stage_{stage_idx}_head_layers.pth")
        if not os.path.exists(ckpt_path):
            # Try pattern for backbone unfreezing
            ckpt_path = os.path.join(out_dir, f"stage_{stage_idx}_unfreeze_*.pth")
            # Check if any file matches this pattern
            import glob
            matches = glob.glob(ckpt_path)
            if not matches:
                break
            ckpt_path = matches[0]
        
        if os.path.exists(ckpt_path):
            stage_num = stage_idx
            last_ckpt = ckpt_path
            stage_idx += 1
        else:
            break
    
    return stage_num, last_ckpt


# ==================== OUTPUT PATHS ====================
os.makedirs(OUT_DIR, exist_ok=True)
checkpoint_paths = []

# ==================== CHECK FOR RESUMPTION ====================
print("\n" + "="*70)
print("CHECKING FOR EXISTING CHECKPOINTS")
print("="*70)

last_completed_stage, last_ckpt = find_last_completed_stage(OUT_DIR)

if last_ckpt is not None:
    print(f"Found checkpoint from Stage {last_completed_stage}: {os.path.basename(last_ckpt)}")
    print(f"RESUMING from Stage {last_completed_stage + 1}")
    resume_from_stage = last_completed_stage + 1
else:
    print("No existing checkpoints found")
    print("STARTING FROM STAGE 0")
    resume_from_stage = 0

# ==================== PROGRESSIVE FINE-TUNING ====================
print("\n" + "="*70)
print("PROGRESSIVE FINE-TUNING STRATEGY")
print("="*70)
print("Stage 0: Classifier head only (frozen backbone)")
print("Stage 1: Classifier + image_fc + text_fc (frozen backbone)")
print("Stage 2: + Last 1 backbone block")
print("Stage 3: + Last 2 backbone blocks")
print("Stage 4: + Last 3 backbone blocks")
print("... (continue unfreezing backbone blocks one by one)")
print("="*70 + "\n")

model = EfficientNetV2MMultimodalClassifier(
    VOCAB_SIZE, len(CLASS_NAMES), train_backbone=False
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device),
    label_smoothing=LABEL_SMOOTHING
)

best_acc = 0.0
backbone_blocks = get_backbone_blocks(model)
num_backbone_blocks = len(backbone_blocks)

print(f"Total backbone blocks: {num_backbone_blocks}\n")

# =====================================================
# STAGE 0: CLASSIFIER HEAD ONLY
# =====================================================
if resume_from_stage <= 0:
    print("="*70)
    print("STAGE 0: Training Classifier Head Only (Backbone Frozen)")
    print("="*70)

    # Freeze everything
    set_requires_grad(model.image_features, False)
    set_requires_grad(model.image_fc, False)
    set_requires_grad(model.text_fc, False)
    set_requires_grad(model.classifier, True)

    print(f"Trainable parameters: {count_trainable_params(model)}")
    print_trainable_layers(model)

    optimizer = make_optimizer(model, lr=1e-3, weight_decay=0.01)
    ckpt_path = os.path.join(OUT_DIR, "stage_0_classifier_only.pth")
    checkpoint_paths.append(ckpt_path)

    # Check if checkpoint exists for this stage
    if os.path.exists(ckpt_path):
        print(f"\nFound existing Stage 0 checkpoint, loading...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        # Try to load best_acc from summary if available
        best_acc = 0.0
    else:
        history, best_acc = train_model(
            model, dataloaders, criterion, optimizer,
            epochs=5, device=device, save_path=ckpt_path,
            freeze_bn_stats=False, 
            stage_name="Stage 0: Classifier Only",
            best_acc_so_far=best_acc
        )
        save_stage_curves(history, OUT_DIR, "stage_0_classifier_only")

    # Load best checkpoint for next stage
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
else:
    print("Skipping Stage 0 (already completed)")
    ckpt_path = os.path.join(OUT_DIR, "stage_0_classifier_only.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    checkpoint_paths.append(ckpt_path)

# =====================================================
# STAGE 1: CLASSIFIER + IMAGE_FC + TEXT_FC
# =====================================================
if resume_from_stage <= 1:
    print("\n" + "="*70)
    print("STAGE 1: Training Head Layers (image_fc + text_fc + classifier)")
    print("="*70)

    set_requires_grad(model.image_features, False)
    set_requires_grad(model.image_fc, True)
    set_requires_grad(model.text_fc, True)
    set_requires_grad(model.classifier, True)

    print(f"Trainable parameters: {count_trainable_params(model)}")
    print_trainable_layers(model)

    optimizer = make_optimizer(model, lr=2e-4, weight_decay=0.01)
    ckpt_path = os.path.join(OUT_DIR, "stage_1_head_layers.pth")
    checkpoint_paths.append(ckpt_path)

    # Check if checkpoint exists for this stage
    if os.path.exists(ckpt_path):
        print(f"\nFound existing Stage 1 checkpoint, loading...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        history, best_acc = train_model(
            model, dataloaders, criterion, optimizer,
            epochs=4, device=device, save_path=ckpt_path,
            freeze_bn_stats=False, 
            stage_name="Stage 1: Head Layers",
            best_acc_so_far=best_acc
        )
        save_stage_curves(history, OUT_DIR, "stage_1_head_layers")

    # Load best checkpoint for next stage
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
else:
    print("Skipping Stage 1 (already completed)")
    ckpt_path = os.path.join(OUT_DIR, "stage_1_head_layers.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    checkpoint_paths.append(ckpt_path)

# =====================================================
# STAGES 2+: PROGRESSIVELY UNFREEZE BACKBONE BLOCKS
# =====================================================
for block_idx in range(1, num_backbone_blocks + 1):
    stage_num = block_idx + 1  # Stage 2 = 1 block, Stage 3 = 2 blocks, etc.
    
    if resume_from_stage <= stage_num:
        print("\n" + "="*70)
        print(f"STAGE {stage_num}: Unfreezing Last {block_idx} Backbone Block(s)")
        print("="*70)

        # Freeze entire backbone first
        set_requires_grad(model.image_features, False)

        # Unfreeze last block_idx blocks
        start_idx = num_backbone_blocks - block_idx
        for i in range(start_idx, num_backbone_blocks):
            set_requires_grad(backbone_blocks[i], True)

        # Keep head trainable
        set_requires_grad(model.image_fc, True)
        set_requires_grad(model.text_fc, True)
        set_requires_grad(model.classifier, True)

        print(f"Trainable parameters: {count_trainable_params(model)}")
        print_trainable_layers(model)

        # Learning rates decrease as we unfreeze more
        lr_backbone = max(1e-6, 1e-5 / (block_idx ** 0.5))
        lr_head = max(5e-6, 5e-5 / (block_idx ** 0.5))

        # Create optimizer with differential learning rates
        backbone_params = []
        head_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("image_features."):
                backbone_params.append(p)
            else:
                head_params.append(p)

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr_backbone})
        if head_params:
            param_groups.append({"params": head_params, "lr": lr_head})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        ckpt_path = os.path.join(OUT_DIR, f"stage_{stage_num}_unfreeze_{block_idx}_blocks.pth")
        checkpoint_paths.append(ckpt_path)

        # Check if checkpoint exists for this stage
        if os.path.exists(ckpt_path):
            print(f"\n✓ Found existing Stage {stage_num} checkpoint, loading...")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            # Reduce epochs as we go deeper
            epochs = max(3, 6 - block_idx // 2)

            history, best_acc = train_model(
                model, dataloaders, criterion, optimizer,
                epochs=epochs, device=device, save_path=ckpt_path,
                freeze_bn_stats=True,
                stage_name=f"Stage {stage_num}: Last {block_idx} Block(s)",
                best_acc_so_far=best_acc
            )
            save_stage_curves(history, OUT_DIR, f"stage_{stage_num}_unfreeze_{block_idx}_blocks")

        # Load best checkpoint for next stage
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Stop if we've unfrozen all blocks
        if block_idx >= num_backbone_blocks:
            print("\nAll backbone blocks unfrozen!")
            break
    else:
        print(f"\nSkipping Stage {stage_num} (already completed)")
        ckpt_path = os.path.join(OUT_DIR, f"stage_{stage_num}_unfreeze_{block_idx}_blocks.pth")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            checkpoint_paths.append(ckpt_path)


# ==================== FINAL EVALUATION ====================
print("\n" + "="*70)
print("FINAL EVALUATION ON TEST SET")
print("="*70)

# Find the best checkpoint (last one that exists)
best_ckpt = None
for ckpt in reversed(checkpoint_paths):
    if os.path.exists(ckpt):
        best_ckpt = ckpt
        break

if best_ckpt is None:
    # Fallback: look for any checkpoint
    import glob
    ckpts = glob.glob(os.path.join(OUT_DIR, "stage_*.pth"))
    if ckpts:
        best_ckpt = max(ckpts, key=os.path.getctime)  # Most recent

print(f"\nLoading best model from: {best_ckpt}")
print("Exists:", os.path.exists(best_ckpt) if best_ckpt else False)

model = EfficientNetV2MMultimodalClassifier(
    vocab_size=VOCAB_SIZE, num_classes=len(CLASS_NAMES)
).to(device)
model.load_state_dict(torch.load(best_ckpt, map_location=device))
model.eval()

test_loader = dataloaders["test"]

all_preds, all_labels, all_paths, all_texts = [], [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating test set"):
        imgs = batch["image"].to(device)
        txts = batch["text_vec"].to(device)
        labels = batch["label"].to(device)

        outputs = model(imgs, txts)
        predicted = outputs.argmax(dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_paths.extend(batch["path"])
        all_texts.extend(batch["text"])

accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).mean()
weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print(f"\n{'='*70}")
print(f"Accuracy on test set: {accuracy:.2f}%")
print(f"Weighted F1-Score: {weighted_f1:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"{'='*70}\n")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar_kws={'label': 'Count'})
plt.title("Confusion Matrix (Test Set)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
plt.close()

per_class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100
print("\nPer-class accuracy:")
for name, acc in zip(CLASS_NAMES, per_class_accuracy):
    print(f"  {name}: {acc:.2f}%")

wandb.run.summary["test_accuracy"] = accuracy
wandb.run.summary["test_weighted_f1"] = weighted_f1

# Log Confusion Matrix
wandb.log({
    "final_confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels,
        preds=all_preds,
        class_names=CLASS_NAMES
    )
})

# Close the W&B run
wandb.finish()

# ==================== MISCLASSIFIED EXAMPLES ====================
misclassified = {name: [] for name in CLASS_NAMES}

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

for i, (y, p) in enumerate(zip(all_labels, all_preds)):
    if y != p:
        true_name = CLASS_NAMES[y]
        pred_name = CLASS_NAMES[p]

        img = Image.open(all_paths[i]).convert("RGB")
        img = transform["test"](img).cpu().numpy().transpose(1, 2, 0)
        img = (img * std) + mean
        img = np.clip(img, 0, 1)

        misclassified[true_name].append({
            "image": img,
            "true": true_name,
            "pred": pred_name,
            "text": all_texts[i]
        })

plt.figure(figsize=(15, 12))
rows = len(CLASS_NAMES)
for row, cname in enumerate(CLASS_NAMES):
    examples = misclassified[cname]
    if len(examples) == 0:
        continue
    selected = random.sample(examples, min(3, len(examples)))
    for col, ex in enumerate(selected):
        plt.subplot(rows, 3, row*3 + col + 1)
        plt.imshow(ex["image"])
        plt.title(f"True: {ex['true']}\nPred: {ex['pred']}\n{ex['text'][:20]}")
        plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "misclassified_examples.png"), dpi=200, bbox_inches="tight")
plt.close()

for cname in CLASS_NAMES:
    print(f"{cname}: {len(misclassified[cname])} misclassified examples")


# ==================== SAVE PREDICTIONS ====================
import pandas as pd

df = pd.DataFrame({
    "path": all_paths,
    "text": all_texts,
    "true": [CLASS_NAMES[i] for i in all_labels],
    "pred": [CLASS_NAMES[i] for i in all_preds],
    "correct": [all_labels[i] == all_preds[i] for i in range(len(all_labels))],
})
csv_path = os.path.join(OUT_DIR, "test_predictions.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved predictions CSV: {csv_path}")

# ==================== SAVE CHECKPOINT SUMMARY ====================
summary_path = os.path.join(OUT_DIR, "training_summary.txt")
with open(summary_path, "w") as f:
    f.write("PROGRESSIVE FINE-TUNING SUMMARY\n")
    f.write("="*70 + "\n\n")
    f.write("Training Stages:\n")
    f.write("  Stage 0: Classifier head only (frozen backbone)\n")
    f.write("  Stage 1: Head layers (image_fc + text_fc + classifier)\n")
    f.write(f"  Stage 2-{len(checkpoint_paths)-1}: Progressive backbone unfreezing (1 block at a time)\n\n")
    f.write("Checkpoints:\n")
    for i, ckpt in enumerate(checkpoint_paths):
        status = "✓" if os.path.exists(ckpt) else "✗"
        f.write(f"  Stage {i}: {status} {os.path.basename(ckpt)}\n")
    f.write(f"\nBest checkpoint: {os.path.basename(best_ckpt) if best_ckpt else 'N/A'}\n\n")
    f.write("Final Metrics:\n")
    f.write(f"  Test Accuracy: {accuracy:.2f}%\n")
    f.write(f"  Weighted F1-Score: {weighted_f1:.4f}\n")
    f.write(f"  Macro F1-Score: {macro_f1:.4f}\n")

print(f"Saved summary: {summary_path}")
print("\nDONE. All outputs saved to:", OUT_DIR)