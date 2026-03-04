import torch
import random
import numpy as np
from torchvision import transforms

from scipy.ndimage import gaussian_filter

class RandomColorTemperature:
    def __init__(self, strength=0.1, p=0.5):
        self.strength = strength
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        shift = random.uniform(-self.strength, self.strength)
        r_scale = 1.0 + shift
        b_scale = 1.0 - shift

        img = img.clone()
        img[0] *= r_scale  # Red channel
        img[2] *= b_scale  # Blue channel

        return torch.clamp(img, 0, 1)


class AddGaussianNoise:
    def __init__(self, mean=0.0, std_range=(0.01, 0.05), p=0.5):
        self.mean = mean
        self.std_range = std_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(img) * std
        img = img + noise
        return torch.clamp(img, 0, 1)


class RandomDownsample:
    def __init__(self, scale_range=(0.5, 0.9), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        _, h, w = img.shape
        scale = random.uniform(*self.scale_range)

        new_h = int(h * scale)
        new_w = int(w * scale)

        img_small = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )

        img_restored = torch.nn.functional.interpolate(
            img_small,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )

        return img_restored.squeeze(0)


class RandomCutout:
    # Random erasing/cutout for occlusion robustness.
    def __init__(self, scale_range=(0.02, 0.2), ratio_range=(0.3, 3.0), p=0.5):
        self.scale_range = scale_range
        self.ratio_range = ratio_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        c, h, w = img.shape
        img = img.clone()
        
        area = h * w
        erase_area = random.uniform(*self.scale_range) * area
        aspect = random.uniform(*self.ratio_range)
        
        erase_h = int(np.sqrt(erase_area / aspect))
        erase_w = int(np.sqrt(erase_area * aspect))
        
        erase_h = min(erase_h, h)
        erase_w = min(erase_w, w)
        
        x = random.randint(0, w - erase_w)
        y = random.randint(0, h - erase_h)
        
        # Random value or mean
        erase_value = random.choice([torch.tensor(0.0), img.mean()])
        img[:, y:y+erase_h, x:x+erase_w] = erase_value
        
        return img


class GarbageSpecificAugmentation:
    # Domain-specific augmentation for garbage/waste classification.
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        c, h, w = img.shape
        img = img.clone()
        
        # Simulate blur/dirt on surface (garbage often has unclear edges)
        if random.random() < 0.5:
            kernel_size = random.choice([3, 5])
            img_np = img.numpy().transpose(1, 2, 0)
            img_np = gaussian_filter(img_np, sigma=random.uniform(0.5, 1.5))
            img = torch.from_numpy(img_np.transpose(2, 0, 1))
        
        return torch.clamp(img, 0, 1)


class MixUp:
    # Mixup augmentation for batch mixing.
    def __init__(self, alpha=1.0, p=0.5):
        self.alpha = alpha
        self.p = p

    def __call__(self, batch_images):
        """
        Apply to batch: batch_images shape [B, C, H, W]
        Returns mixed batch and lambda weights.
        """
        if random.random() > self.p:
            return batch_images, torch.ones(len(batch_images))
        
        lam = np.random.beta(self.alpha, self.alpha)
        b = len(batch_images)
        index = torch.randperm(b)
        mixed = lam * batch_images + (1 - lam) * batch_images[index]
        
        return mixed, torch.full((b,), lam)