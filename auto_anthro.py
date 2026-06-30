"""Estimate 17 CIPIC anthropometric measurements from two iPhone photos.

Replaces the interactive `AnthropomorphicFeatures.ipynb` workflow:

  Photo of you holding an 8.5x11 in. page ─► auto-detect page corners
                                          ─► perspective-rectify
                                          ─► known pixels-per-inch scale
                                          ─► click landmarks ─► measurements

The *page detection + rectification* step is fully automatic here (vs. the
original notebook which needed four clicks per photo). The *landmark*
clicks remain interactive — automated pinna-landmark detection is a real
research problem and requires a trained landmark model we don't have. The
existing literature uses MediaPipe FaceMesh for head landmarks and custom
pinna landmark models for ear-specific ones; see the README for the path
forward.

Even with manual landmark clicks, this script removes the notebook /
ipywidgets dependency entirely — it works in any plain Python environment.

Usage:
  python auto_anthro.py --front front.jpg --side side.jpg --output anthro.txt
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Page is 8.5 x 11 inches; rectify to this many pixels per inch so that the
# whole rectified region is 850 x 1100 px (matches the original notebook's
# convention, which made 1 px == 0.01 in == 2.54/100 cm).
PX_PER_INCH = 100
PAGE_WIDTH_IN = 8.5
PAGE_HEIGHT_IN = 11.0
PAGE_WIDTH_PX = int(PAGE_WIDTH_IN * PX_PER_INCH)   # 850
PAGE_HEIGHT_PX = int(PAGE_HEIGHT_IN * PX_PER_INCH)  # 1100
CM_PER_PIXEL = 2.54 / PX_PER_INCH


@dataclass
class RectifiedPhoto:
    image: np.ndarray         # rectified BGR/RGB image
    page_corners: np.ndarray  # 4x2, in original-image coords, TL/TR/BR/BL


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return 4 points ordered as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.stack([
        pts[np.argmin(s)],  # TL: smallest x+y
        pts[np.argmin(d)],  # TR: smallest y-x
        pts[np.argmax(s)],  # BR: largest x+y
        pts[np.argmax(d)],  # BL: largest y-x
    ], axis=0)


def detect_page_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """Find the 8.5x11 page in a photo.

    Returns 4x2 array of (x, y) corners in TL/TR/BR/BL order, or None if no
    plausible page contour was found.

    Strategy: edge map -> contour finding -> filter for convex 4-vertex
    polygons -> pick the one closest to the known 8.5:11 aspect ratio,
    weighted by area. This handles the iPhone use case where the page is
    held in front of a body / wall: the page is bright, has straight edges,
    and is roughly the right aspect ratio.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Close gaps in the edge map so partial page outlines still register.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    target_aspect = PAGE_WIDTH_IN / PAGE_HEIGHT_IN  # 8.5/11 ~ 0.773
    img_area = image.shape[0] * image.shape[1]
    candidates: List[Tuple[float, np.ndarray]] = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        if area < 0.01 * img_area:  # reject tiny squares
            continue
        ordered = order_corners(approx.reshape(4, 2))
        w_top = np.linalg.norm(ordered[1] - ordered[0])
        w_bot = np.linalg.norm(ordered[2] - ordered[3])
        h_l = np.linalg.norm(ordered[3] - ordered[0])
        h_r = np.linalg.norm(ordered[2] - ordered[1])
        width = (w_top + w_bot) / 2
        height = (h_l + h_r) / 2
        if height < 1 or width < 1:
            continue
        aspect = width / height
        # Score: high area, aspect close to 8.5:11 (in either orientation).
        aspect_err = min(abs(aspect - target_aspect),
                         abs(aspect - 1.0 / target_aspect))
        score = area * (1.0 / (1.0 + 5.0 * aspect_err))
        candidates.append((score, ordered))

    if not candidates:
        return None
    candidates.sort(key=lambda kv: -kv[0])
    return candidates[0][1]


def rectify_to_page(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Apply a 4-point perspective transform so the page fills a canonical
    PAGE_WIDTH_PX x PAGE_HEIGHT_PX rectangle in the output image."""
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    dst = np.array([
        [0, 0],
        [PAGE_WIDTH_PX - 1, 0],
        [PAGE_WIDTH_PX - 1, PAGE_HEIGHT_PX - 1],
        [0, PAGE_HEIGHT_PX - 1],
    ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, H, (PAGE_WIDTH_PX, PAGE_HEIGHT_PX))


def auto_rectify(image: np.ndarray) -> RectifiedPhoto:
    corners = detect_page_corners(image)
    if corners is None:
        raise RuntimeError(
            "Could not auto-detect an 8.5x11 page in the photo. Make sure "
            "the page is fully visible, well-lit, held flat, and roughly "
            "parallel to the camera plane."
        )
    rectified = rectify_to_page(image, corners)
    return RectifiedPhoto(image=rectified, page_corners=corners)


# ---------------- measurements ----------------

def distance_cm(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    return float(np.linalg.norm(p1 - p2) * CM_PER_PIXEL)


# Each entry: (index, label, photo_key, kind)
# kind="pair" -> click two landmark points, distance between them
# kind="from_center" -> click one point, distance from the head-center reference
# kind="prompt" -> typed-in number (height etc.)
MEASUREMENT_SPEC = [
    (0,  "x1  head width",          "front", "pair"),
    (1,  "x2  head height",         "side",  "pair"),
    (2,  "x3  head depth",          "side",  "pair"),
    (3,  "x4  pinna offset down",   "side",  "from_center"),
    (4,  "x5  pinna offset back",   "side",  "from_center"),
    (5,  "x6  neck width",          "front", "pair"),
    (6,  "x7  neck height",         "side",  "pair"),
    (7,  "x8  neck depth",          "side",  "pair"),
    (8,  "x9  torso top width",     "front", "pair"),
    (9,  "x10 torso top height",    "side",  "pair"),
    (10, "x11 torso top depth",     "side",  "pair"),
    (11, "x12 shoulder width",      "front", "pair"),
    (12, "x13 head offset forward", "side",  "from_center"),
    (13, "x14 standing height",     None,    "prompt"),
    (14, "x15 seated height",       None,    "prompt"),
    (15, "x16 head circumference",  None,    "prompt"),
    (16, "x17 shoulder circumference", None, "prompt"),
]


def _click_points(image: np.ndarray, n: int, prompt: str) -> List[Tuple[float, float]]:
    """Show image and collect n clicks via matplotlib's ginput."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 12))
    plt.imshow(image)
    plt.title(prompt)
    plt.tight_layout()
    pts = plt.ginput(n, show_clicks=True, timeout=0)
    plt.close()
    return pts


def collect_measurements_interactive(
    rectified: dict, head_center_side: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """rectified: {'front': RectifiedPhoto, 'side': RectifiedPhoto}."""
    anthro = np.zeros(17, dtype=np.float64)

    # Click head center on the side photo once if not provided.
    if head_center_side is None:
        head_center_side = _click_points(
            rectified["side"].image, 1, "Click the center of the head (one point)"
        )[0]

    for idx, label, key, kind in MEASUREMENT_SPEC:
        if kind == "pair":
            pts = _click_points(rectified[key].image, 2,
                                f"{label}: click two endpoints")
            anthro[idx] = distance_cm(pts[0], pts[1])
        elif kind == "from_center":
            pts = _click_points(rectified[key].image, 1,
                                f"{label}: click landmark (will measure from head center)")
            anthro[idx] = distance_cm(head_center_side, pts[0])
        elif kind == "prompt":
            while True:
                raw = input(f"{label} (cm): ").strip()
                try:
                    anthro[idx] = float(raw)
                    break
                except ValueError:
                    print("  please enter a number")

    return anthro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", type=Path, required=True,
                        help="Front-facing photo with 8.5x11 page held in view")
    parser.add_argument("--side", type=Path, required=True,
                        help="Side-facing photo with 8.5x11 page held in view")
    parser.add_argument("--output", type=Path, default=Path("anthro.txt"),
                        help="Where to write 17 comma-separated cm values")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Only auto-detect + rectify; skip landmark clicks "
                             "(use for testing the page-detection step alone)")
    args = parser.parse_args()

    front_img = cv2.cvtColor(cv2.imread(str(args.front)), cv2.COLOR_BGR2RGB)
    side_img = cv2.cvtColor(cv2.imread(str(args.side)), cv2.COLOR_BGR2RGB)

    print("Detecting page in front photo...")
    front_rect = auto_rectify(front_img)
    print(f"  page corners: {front_rect.page_corners.tolist()}")

    print("Detecting page in side photo...")
    side_rect = auto_rectify(side_img)
    print(f"  page corners: {side_rect.page_corners.tolist()}")

    if args.non_interactive:
        print("Skipping interactive landmark collection (--non-interactive).")
        return

    anthro = collect_measurements_interactive(
        {"front": front_rect, "side": side_rect}
    )
    np.savetxt(str(args.output), anthro.reshape(1, -1), delimiter=",", fmt="%.3f")
    print(f"Wrote {len(anthro)} measurements (cm) to {args.output}")


if __name__ == "__main__":
    main()
