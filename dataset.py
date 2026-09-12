import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CSVDataset(Dataset):
    """
    Dataset reading from CSV files with the following column layout:
        col 0: Image path
        col 1: Sample ID
        col 2: Question type (qtype) - used to log and analyze expert utilization
        col 3: Question text
        col 4-7: Multiple-choice options (A, B, C, D)
        col 8: Reserved / Metadata
        col 9: Ground truth answer
    """

    def __init__(self, csv_path: str, image_root: str = None):
        """
        Args:
            csv_path: Path to the annotation CSV file.
            image_root: Optional directory containing images if paths in CSV are relative.
        """
        self.data = pd.read_csv(csv_path, header=None, low_memory=False)
        self.image_root = image_root

        if self.data.shape[1] < 10:
            raise ValueError(
                f"CSV file must have >= 10 columns, but found {self.data.shape[1]} columns."
            )

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"✅ Loaded dataset: {len(self.data)} samples from {csv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ── Image Loading ──
        img_path = str(row[0])

        if self.image_root and not os.path.isfile(img_path):
            candidate = os.path.join(self.image_root, os.path.basename(img_path))
            if os.path.isfile(candidate):
                img_path = candidate

        # Fallback for dataset paths recorded with prefix differences
        if not os.path.isfile(img_path):
            fallback_path = img_path.replace("val", "C:/Dataset", 1).replace("/", "\\")
            if os.path.isfile(fallback_path):
                img_path = fallback_path

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image at {img_path}: {e}")

        image = self.transform(image)

        # ── Text Loading ──
        question = str(row[3])
        answer = str(row[9]).strip()
        qtype = str(row[2]).strip()   # Question type identifier (e.g., HDC, DC, SI, etc.)

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "qtype": qtype,
        }


def collate_fn(batch):
    """
    Collate function to batch multimodal samples together.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    questions = [b["question"] for b in batch]
    answers = [b["answer"] for b in batch]
    qtypes = [b["qtype"] for b in batch]
    return images, questions, answers, qtypes
