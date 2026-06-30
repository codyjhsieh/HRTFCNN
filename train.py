"""HRTF CNN training script.

Trains a small image+anthropometry network on the CIPIC dataset to predict
head-related impulse responses (HRIRs) for arbitrary (azimuth, elevation)
directions. Designed to fit comfortably on a free T4 GPU (16 GB) on
Lightning AI: frozen ImageNet backbone + small MLPs, fp16 mixed precision,
batch 64.

Expected data layout (relative to --data-dir, default ./data):
  cipic_hrtf_sofa/subject_XXX.sofa
  CIPIC_hrtf_database/anthropometry/anthro.mat
  ear_photos/Subject_XXX/<photo>.jpg

SOFA files: wget -r -l1 -np http://sofacoustics.org/data/database/cipic/
Anthropometry + ear photos: download CIPIC_hrtf_database.zip from
  https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/

Usage:
  pip install -r requirements.txt
  python train.py --data-dir ./data --epochs 50 --batch-size 64
"""

import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import scipy.io
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.dsp import (
    ir_to_log_magnitude_torch,
    log_spectral_distance_torch,
)
from utils.hrtf import get_hrtf_sofa

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
N_TAPS = 200
N_FFT = 256
N_MAG_BINS = N_FFT // 2 + 1  # 129
N_DIRECTIONS = 1250


def find_ear_image(ear_photos_dir: Path, subject_num: int):
    num = f"{subject_num:03d}"
    subj_dir = ear_photos_dir / f"Subject_{num}"
    if not subj_dir.exists():
        return None, None
    for ear_idx, key in [(0, "left"), (1, "right")]:
        for p in sorted(subj_dir.iterdir()):
            name = p.name.lower()
            if key in name and name.endswith((".jpg", ".jpeg", ".png")):
                return p, ear_idx
    return None, None


def build_subject_index(data_dir: Path, max_subject: int = 200):
    sofa_dir = data_dir / "cipic_hrtf_sofa"
    photos_dir = data_dir / "ear_photos"
    anthro_path = data_dir / "CIPIC_hrtf_database" / "anthropometry" / "anthro.mat"

    if not anthro_path.exists():
        raise FileNotFoundError(f"Missing anthropometry file: {anthro_path}")

    anthro = np.asarray(scipy.io.loadmat(str(anthro_path))["X"])

    entries = []
    az_ref = None
    el_ref = None
    for subj in range(max_subject + 1):
        sofa_path = sofa_dir / f"subject_{subj:03d}.sofa"
        if not sofa_path.exists():
            continue
        if subj >= len(anthro) or np.isnan(anthro[subj]).any():
            continue
        img_path, ear_idx = find_ear_image(photos_dir, subj)
        if img_path is None:
            continue
        try:
            hrtf = get_hrtf_sofa(str(sofa_dir) + "/", subj)
        except Exception as err:
            print(f"Skip subject {subj}: SOFA load failed: {err}")
            continue
        if az_ref is None:
            az_ref = hrtf.azimuths.astype(np.float32)
            el_ref = hrtf.elevations.astype(np.float32)
        entries.append({
            "subject": subj,
            "anthro": anthro[subj].astype(np.float32),
            "image_path": str(img_path),
            "ear": ear_idx,
            "impulses": hrtf.impulses[:, ear_idx, :].astype(np.float32),
        })

    if not entries:
        raise RuntimeError("No usable subjects found — check data layout.")
    print(f"Usable subjects: {len(entries)}")
    return entries, az_ref, el_ref


class CipicHRTFDataset(Dataset):
    def __init__(self, subject_entries, azimuths, elevations, transform):
        self.entries = subject_entries
        self.azimuths = azimuths
        self.elevations = elevations
        self.transform = transform
        # Flatten: (subject_idx, direction_idx)
        self.index = [
            (s_idx, d_idx)
            for s_idx in range(len(subject_entries))
            for d_idx in range(len(azimuths))
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s_idx, d_idx = self.index[i]
        entry = self.entries[s_idx]
        img = Image.open(entry["image_path"]).convert("RGB")
        image = self.transform(img)
        anthro = torch.from_numpy(entry["anthro"])
        direction = torch.tensor(
            [self.azimuths[d_idx], self.elevations[d_idx]], dtype=torch.float32
        )
        target = torch.from_numpy(entry["impulses"][d_idx])
        return image, anthro, direction, target


def make_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class HRTFModel(pl.LightningModule):
    def __init__(
        self,
        anthro_dim: int,
        backbone_name: str = "resnet18",
        image_feat_dim: int = 64,
        anthro_feat_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool="avg"
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        backbone_out = self.backbone.num_features

        self.image_head = nn.Sequential(
            nn.Linear(backbone_out, 128), nn.ReLU(),
            nn.Linear(128, image_feat_dim), nn.ReLU(),
        )
        self.anthro_head = nn.Sequential(
            nn.Linear(anthro_dim + 2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, anthro_feat_dim), nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(image_feat_dim + anthro_feat_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, N_MAG_BINS),
        )

    def forward(self, image, anthro, direction):
        with torch.no_grad():
            img_feat = self.backbone(image)
        img_feat = self.image_head(img_feat)
        tab = torch.cat([anthro, direction], dim=1)
        tab_feat = self.anthro_head(tab)
        z = torch.cat([img_feat, tab_feat], dim=1)
        return self.regressor(z)

    def _step(self, batch, stage):
        image, anthro, direction, target_ir = batch
        pred_log_mag = self(image, anthro, direction)
        target_log_mag = ir_to_log_magnitude_torch(target_ir, n_fft=N_FFT)
        # MSE on log-magnitude == LSD^2 averaged over freq — same objective,
        # logged in both forms for readability.
        mse = F.mse_loss(pred_log_mag, target_log_mag)
        lsd = log_spectral_distance_torch(pred_log_mag, target_log_mag)
        self.log(f"{stage}/log_mag_mse", mse, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/lsd_db", lsd, prog_bar=True, sync_dist=True)
        return mse

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    entries, azimuths, elevations = build_subject_index(args.data_dir)
    anthro_dim = len(entries[0]["anthro"])

    indices = np.arange(len(entries))
    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_frac, random_state=args.seed
    )
    train_entries = [entries[i] for i in train_idx]
    val_entries = [entries[i] for i in val_idx]
    print(f"Train subjects: {len(train_entries)}  Val subjects: {len(val_entries)}")

    train_ds = CipicHRTFDataset(train_entries, azimuths, elevations,
                                make_transforms(train=True))
    val_ds = CipicHRTFDataset(val_entries, azimuths, elevations,
                              make_transforms(train=False))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    model = HRTFModel(
        anthro_dim=anthro_dim,
        backbone_name=args.backbone,
        lr=args.lr,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=str(args.output_dir),
        filename="hrtf-epoch{epoch:02d}",
        monitor="val/lsd_db",
        mode="min",
        save_top_k=2,
        auto_insert_metric_name=False,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        callbacks=[ckpt],
        log_every_n_steps=25,
        default_root_dir=str(args.output_dir),
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"Best checkpoint: {ckpt.best_model_path}  LSD={ckpt.best_model_score:.3f} dB")


if __name__ == "__main__":
    main()
