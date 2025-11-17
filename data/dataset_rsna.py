"""RSNA Pneumonia Detection Challenge Dataset loader."""

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class RSNADataset(Dataset):
    """RSNA dataset with metadata CSV for SSL pretraining."""
    def __init__(self, image_dir, csv_path, transform=None, ssl_method='fedcon'):
        self.image_dir = image_dir
        self.transform = transform
        self.ssl_method = ssl_method  # 'fedcon' returns 2 views, 'fedmae' returns 1
        self.metadata = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        patient_id = row['patientId']
        # Reference code uses .png files (preprocessed from DICOM)
        img_path = os.path.join(self.image_dir, f"{patient_id}.png")
        
        if not os.path.exists(img_path):
            # Fallback to .dcm if .png doesn't exist
            img_path = os.path.join(self.image_dir, f"{patient_id}.dcm")
            try:
                import pydicom
                dcm = pydicom.dcmread(img_path)
                image = Image.fromarray(dcm.pixel_array).convert("RGB")
            except:
                raise FileNotFoundError(f"Could not find image: {patient_id}.png or {patient_id}.dcm")
        else:
            image = Image.open(img_path).convert("RGB")
        
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


def get_rsna_dataset(image_dir, csv_path, transform=None, ssl_method='fedcon'):
    """Load RSNA dataset."""
    return RSNADataset(image_dir=image_dir, csv_path=csv_path, transform=transform, ssl_method=ssl_method)
