"""Kaggle Pneumonia Dataset loader."""

import os
from torch.utils.data import Dataset
from PIL import Image


class KagglePneumoniaDataset(Dataset):
    """Kaggle Pneumonia dataset with two image views for contrastive learning."""
    def __init__(self, root_dir, transform=None, ssl_method='fedcon'):
        self.root_dir = root_dir
        self.transform = transform
        self.ssl_method = ssl_method  # 'fedcon' returns 2 views, 'fedmae' returns 1
        self.image_files = []
        # Combine NORMAL and PNEUMONIA for unsupervised pretraining
        for subdir in ["NORMAL", "PNEUMONIA"]:
            d = os.path.join(root_dir, subdir)
            if os.path.isdir(d):
                self.image_files.extend([
                    os.path.join(d, f) for f in os.listdir(d) 
                    if f.endswith(('.jpeg', '.jpg', '.png'))
                ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform:
            if self.ssl_method == 'fedcon':
                # Contrastive: return two augmented views
                view1 = self.transform(image)
                view2 = self.transform(image)
                return view1, view2
            else:
                # MAE: return single image
                return self.transform(image)
        return image


def get_kaggle_dataset(root_dir, transform=None, ssl_method='fedcon'):
    """Load Kaggle dataset."""
    return KagglaPneumoniaDataset(root_dir=root_dir, transform=transform, ssl_method=ssl_method)
