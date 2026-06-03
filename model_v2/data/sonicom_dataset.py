"""SONICOM dataloader: pairs an ear photo + optional anthropometry with target HRTFs.

This module encodes **documented assumptions** about the SONICOM release layout. They
could not be validated against the real dataset in the authoring environment, so each
load-bearing assumption is marked ``# TODO(verify)``. Check them against your actual
SONICOM download and adjust.

What this loader does *not* use: depth photos, 3D scans, headphone transfer functions.
Only the 2D ear photo (input), anthropometry (optional input), and the measured HRTF
(label) are read. See ../README.md "Why no depth".

Target representation per direction:
    log-magnitude spectrum, shape (2, F)   (ear 0 = left, ear 1 = right)
    ITD, scalar in samples (signed; >0 means left ear leads)

HRIR -> (log-magnitude, ITD) preprocessing:
    * estimate ITD per direction by cross-correlating the two ears' HRIRs;
    * time-align each ear (remove the bulk onset delay) so magnitude modeling is clean;
    * rFFT -> magnitude -> natural log (floored to avoid log(0)).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

try:  # SOFA reading: prefer `sofar`, fall back to h5py (SOFA is HDF5/netCDF under the hood)
    import sofar  # type: ignore
    _HAVE_SOFAR = True
except Exception:  # pragma: no cover
    _HAVE_SOFAR = False

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - allow importing for docs without torch
    Dataset = object  # type: ignore


# --- HRIR -> target conversion ------------------------------------------------

def hrir_to_logmag_itd(hrir: np.ndarray, n_fft: int, n_freq: int,
                       eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Convert a (D, 2, T) HRIR block to (log_mag (D,2,F), itd (D,)).

    ITD is estimated via the lag that maximizes cross-correlation between ears.
    """
    d = hrir.shape[0]
    log_mag = np.empty((d, 2, n_freq), dtype=np.float32)
    itd = np.empty((d,), dtype=np.float32)
    for i in range(d):
        left, right = hrir[i, 0], hrir[i, 1]
        itd[i] = _estimate_itd(left, right)
        for ear in (0, 1):
            spec = np.fft.rfft(hrir[i, ear], n=n_fft)
            mag = np.abs(spec)[:n_freq]
            log_mag[i, ear] = np.log(np.maximum(mag, eps))
    return log_mag, itd


def _estimate_itd(left: np.ndarray, right: np.ndarray) -> float:
    """Signed ITD in samples via full cross-correlation argmax (>0: left leads)."""
    corr = np.correlate(left, right, mode="full")
    lag = int(np.argmax(np.abs(corr))) - (len(right) - 1)
    return float(lag)


# --- Dataset ------------------------------------------------------------------

@dataclass
class Subject:
    sid: str
    sofa_path: str
    photo_path: str | None
    anthro: np.ndarray | None  # (anthro_dim,) or None if missing


class SonicomHRTFDataset(Dataset):
    """One item = one subject's full HRTF set + photo + optional anthropometry.

    Returning whole subjects (not individual directions) lets the model encode the
    subject embedding once and decode many directions per forward pass, which matches
    how ``HRTFNet`` is meant to be trained and is far more efficient.
    """

    def __init__(self, root: str, subject_ids: list[str] | None = None,
                 img_size: int = 128, n_fft: int = 256, n_freq: int = 128,
                 anthro_dim: int = 10, max_directions: int | None = None):
        self.root = root
        self.img_size = img_size
        self.n_fft = n_fft
        self.n_freq = n_freq
        self.anthro_dim = anthro_dim
        self.max_directions = max_directions
        self.anthro_table = self._load_anthro_table(root, anthro_dim)
        self.subjects = self._index_subjects(root, subject_ids)
        if not self.subjects:
            raise RuntimeError(
                f"No SONICOM subjects found under {root!r}. Check the path and the "
                f"# TODO(verify) layout assumptions in sonicom_dataset.py.")

    # -- indexing ----------------------------------------------------------

    def _index_subjects(self, root: str, subject_ids) -> list[Subject]:
        # TODO(verify): SONICOM ships per-subject SOFA files; adjust the glob/convention
        # (e.g. FreeFieldCompMinPhase_48kHz) to the variant you intend to train on.
        sofa_glob = os.path.join(root, "**", "*.sofa")
        subjects: list[Subject] = []
        for sofa_path in sorted(glob.glob(sofa_glob, recursive=True)):
            sid = self._subject_id_from_path(sofa_path)
            if subject_ids is not None and sid not in subject_ids:
                continue
            subjects.append(Subject(
                sid=sid,
                sofa_path=sofa_path,
                photo_path=self._find_photo(root, sid),
                anthro=self.anthro_table.get(sid),
            ))
        return subjects

    @staticmethod
    def _subject_id_from_path(path: str) -> str:
        # TODO(verify): SONICOM subject ids look like "P0001". Extract from filename.
        base = os.path.basename(path)
        for tok in base.replace("-", "_").split("_"):
            if tok and tok[0] in "Pp" and tok[1:].isdigit():
                return tok.upper()
        return os.path.splitext(base)[0]

    def _find_photo(self, root: str, sid: str) -> str | None:
        # TODO(verify): point this at SONICOM's 2D ear-photo files for the subject.
        for ext in ("png", "jpg", "jpeg"):
            hits = glob.glob(os.path.join(root, "**", f"*{sid}*ear*.{ext}"),
                             recursive=True)
            if hits:
                return sorted(hits)[0]
        return None

    def _load_anthro_table(self, root: str, anthro_dim: int) -> dict:
        # TODO(verify): SONICOM_anthropometries.csv — one row per ear, keyed by subject.
        import csv
        table: dict[str, np.ndarray] = {}
        hits = glob.glob(os.path.join(root, "**", "*anthropometr*.csv"), recursive=True)
        if not hits:
            return table
        with open(sorted(hits)[0], newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                sid = row[0].strip().upper()
                vals = []
                for x in row[1:]:
                    try:
                        vals.append(float(x))
                    except ValueError:
                        pass
                if vals:
                    v = np.array(vals[:anthro_dim], dtype=np.float32)
                    if v.shape[0] < anthro_dim:
                        v = np.pad(v, (0, anthro_dim - v.shape[0]))
                    table[sid] = v
        return table

    # -- loading -----------------------------------------------------------

    def _read_sofa(self, path: str) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (hrir (D,2,T), directions (D,2) az/el degrees, fs)."""
        if _HAVE_SOFAR:
            s = sofar.read_sofa(path, verify=False)
            ir = np.asarray(s.Data_IR, dtype=np.float32)          # (M, R, N)
            pos = np.asarray(s.SourcePosition, dtype=np.float32)  # (M, 3) az,el,r
            fs = float(np.atleast_1d(s.Data_SamplingRate)[0])
        else:  # h5py fallback
            import h5py
            with h5py.File(path, "r") as h:
                ir = np.asarray(h["Data.IR"], dtype=np.float32)
                pos = np.asarray(h["SourcePosition"], dtype=np.float32)
                fs = float(np.asarray(h["Data.SamplingRate"]).ravel()[0])
        return ir, pos[:, :2], fs

    def _load_photo(self, path: str | None) -> np.ndarray:
        if path is None:
            return np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
        from PIL import Image
        img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))  # HWC -> CHW

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> dict:
        sub = self.subjects[idx]
        hrir, directions, fs = self._read_sofa(sub.sofa_path)
        # SONICOM IR is (M, R, N); we want (D, 2, T).
        if hrir.shape[1] != 2:
            hrir = hrir[:, :2, :]
        if self.max_directions is not None and hrir.shape[0] > self.max_directions:
            sel = np.linspace(0, hrir.shape[0] - 1, self.max_directions).astype(int)
            hrir, directions = hrir[sel], directions[sel]
        log_mag, itd = hrir_to_logmag_itd(hrir, self.n_fft, self.n_freq)

        anthro = sub.anthro
        present = anthro is not None
        if anthro is None:
            anthro = np.zeros((self.anthro_dim,), dtype=np.float32)

        return {
            "sid": sub.sid,
            "image": torch.from_numpy(self._load_photo(sub.photo_path)),
            "anthro": torch.from_numpy(anthro),
            "anthro_present": torch.tensor(float(present)),
            "az": torch.from_numpy(directions[:, 0].astype(np.float32)),
            "el": torch.from_numpy(directions[:, 1].astype(np.float32)),
            "log_mag": torch.from_numpy(log_mag),   # (D, 2, F)
            "itd": torch.from_numpy(itd),           # (D,)
            "fs": torch.tensor(float(fs)),
        }


def collate_subject_batch(batch: list[dict]) -> dict:
    """Collate subjects that may have different direction counts.

    Directions are concatenated across subjects with a ``subject_index`` tensor so the
    training loop can scatter the per-subject embedding to its directions.
    """
    images = torch.stack([b["image"] for b in batch])
    anthro = torch.stack([b["anthro"] for b in batch])
    present = torch.stack([b["anthro_present"] for b in batch])

    az, el, log_mag, itd, subj_idx = [], [], [], [], []
    for i, b in enumerate(batch):
        n = b["az"].shape[0]
        az.append(b["az"]); el.append(b["el"])
        log_mag.append(b["log_mag"]); itd.append(b["itd"])
        subj_idx.append(torch.full((n,), i, dtype=torch.long))
    return {
        "sids": [b["sid"] for b in batch],
        "image": images,
        "anthro": anthro,
        "anthro_present": present,
        "az": torch.cat(az),
        "el": torch.cat(el),
        "log_mag": torch.cat(log_mag),
        "itd": torch.cat(itd),
        "subject_index": torch.cat(subj_idx),
        "fs": float(batch[0]["fs"]),
    }
