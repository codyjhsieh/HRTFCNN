"""Predict a personal HRTF and write it as a CIPIC-compatible SOFA file.

Given a trained checkpoint plus an ear photo and 17 anthropometric numbers
for a new subject, this script runs the model over all 1250 CIPIC directions,
reconstructs a minimum-phase impulse response from the predicted
log-magnitude spectrum, and writes the result as a `.sofa` file that drops
into IRCAM SPAT, the SOFA toolbox, etc.

The model is per-ear (one image, one IR), so for a stereo HRTF you should
run this twice — once with the left-ear photo, once with the right-ear
photo — and merge the two SOFA files. For a first pass, mirroring the
single-ear output is also a defensible approximation.

Usage:
  python predict.py \
      --checkpoint checkpoints/hrtf-epoch49.ckpt \
      --ear-photo path/to/ear.jpg \
      --anthro 17.2,15.8,18.1,...  (17 comma-separated floats) \
      --template data/template.sofa \
      --output predicted.sofa
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from train import (
    HRTFModel,
    N_FFT,
    N_TAPS,
    make_transforms,
)
from utils.dsp import log_magnitude_to_min_phase_ir
from utils.hrtf import CipicHRTF, create_cipic_hrtf


def parse_anthro(arg: str) -> np.ndarray:
    """Either 'a,b,c,...' inline or a path to a .npy / .txt file."""
    p = Path(arg)
    if p.exists():
        if p.suffix == ".npy":
            arr = np.load(p)
        else:
            arr = np.loadtxt(p, delimiter=",")
    else:
        arr = np.array([float(x) for x in arg.split(",")], dtype=np.float64)
    return arr.astype(np.float32).flatten()


def load_template_directions(template_path: Path):
    hrtf = CipicHRTF(str(template_path), 44100.0)
    az = hrtf.azimuths.astype(np.float32)
    el = hrtf.elevations.astype(np.float32)
    return az, el


@torch.no_grad()
def predict_log_magnitude(
    model: HRTFModel,
    image_tensor: torch.Tensor,
    anthro_tensor: torch.Tensor,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Returns log-magnitude in dB, shape (n_directions, n_mag_bins)."""
    model.eval()
    n = len(azimuths)
    outputs: List[np.ndarray] = []

    # Tile the image and anthro across the batch — they don't change with
    # direction. (Tensor.expand avoids the memory cost of a real copy.)
    img_batch_template = image_tensor.unsqueeze(0)
    anthro_batch_template = anthro_tensor.unsqueeze(0)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bsz = end - start
        images = img_batch_template.expand(bsz, -1, -1, -1).to(device)
        anthros = anthro_batch_template.expand(bsz, -1).to(device)
        directions = torch.from_numpy(
            np.stack([azimuths[start:end], elevations[start:end]], axis=1)
        ).to(device)
        log_mag = model(images, anthros, directions)
        outputs.append(log_mag.cpu().numpy())

    return np.concatenate(outputs, axis=0)


def run_inference(
    checkpoint: Path,
    ear_photo: Path,
    anthro: np.ndarray,
    template: Path,
    output: Path,
    device: str = "cpu",
) -> Path:
    """Programmatic entrypoint — used by tests."""
    ckpt = torch.load(str(checkpoint), map_location=device)
    hparams = ckpt["hyper_parameters"]
    model = HRTFModel(
        anthro_dim=hparams["anthro_dim"],
        backbone_name=hparams.get("backbone_name", "resnet18"),
        image_feat_dim=hparams.get("image_feat_dim", 64),
        anthro_feat_dim=hparams.get("anthro_feat_dim", 64),
        lr=hparams.get("lr", 1e-3),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    if anthro.shape[0] != hparams["anthro_dim"]:
        raise ValueError(
            f"--anthro has {anthro.shape[0]} values; checkpoint expects "
            f"{hparams['anthro_dim']}."
        )

    transform = make_transforms(train=False)
    image = transform(Image.open(ear_photo).convert("RGB"))
    anthro_t = torch.from_numpy(anthro)

    azimuths, elevations = load_template_directions(template)

    log_mag = predict_log_magnitude(
        model, image, anthro_t, azimuths, elevations, device=device
    )

    ir = log_magnitude_to_min_phase_ir(
        log_mag, n_fft=N_FFT, n_taps=N_TAPS
    )

    # CIPIC SOFA Data_IR is (n_directions, n_ears, n_taps). The trained
    # model is per-ear; we duplicate into both channels so the SOFA file is
    # well-formed. Run predict.py twice (per ear) and merge for a proper
    # stereo HRTF.
    stereo_ir = np.stack([ir, ir], axis=1)

    # We loaded directions from the template in interaural-polar coords
    # (that's what CipicHRTF returns); create_cipic_hrtf reads them back and
    # writes vertical-polar to SOFA.
    create_cipic_hrtf(
        str(template), str(output),
        stereo_ir, elevations, azimuths,
    )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ear-photo", type=Path, required=True)
    parser.add_argument("--anthro", type=str, required=True,
                        help="Comma-separated floats, or path to .npy/.txt")
    parser.add_argument("--template", type=Path, default=Path("data/template.sofa"))
    parser.add_argument("--output", type=Path, default=Path("predicted.sofa"))
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    anthro = parse_anthro(args.anthro)
    out_path = run_inference(
        checkpoint=args.checkpoint,
        ear_photo=args.ear_photo,
        anthro=anthro,
        template=args.template,
        output=args.output,
        device=args.device,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
