# Fixed-Camera Overhead Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Project YOLO11+BoT-SORT player tracks from a fixed-camera touch rugby clip onto a 2D overhead minimap (video + CSV + trails image), calibrated once per clip via a manual click tool.

**Architecture:** A shared geometry module (`field_calib.py`) owns the 70×50 m field model, homography fitting, and pitch/grid drawing. `calibrate_field.py` is a thin interactive matplotlib UI over it (clicks allowed outside the image bounds for off-frame corners). `project_overhead.py` reuses the tracking loop conventions from `track_humans.py` and adds projection + minimap rendering.

**Tech Stack:** Python 3.12 (Poetry env in `experiments/`), OpenCV, NumPy, matplotlib (+PyQt6 backend), Ultralytics YOLO11 + BoT-SORT, supervision.

**Testing note:** Per the approved spec, `experiments/` is an exploratory gitignored area with **no formal unit tests** — each task instead has explicit headless verification commands (synthetic-data round-trips, end-to-end runs) and the calibration tool bakes in a visual grid-overlay check. `experiments/` is gitignored, so there are no per-task commits; the plan and spec live in `docs/`.

**Coordinate system (used everywhere):** X = 0–50 m across the field (left sideline → right, as seen from the camera), Y = 0 at the camera's score line → 70 at the far score line. Homography `H` maps image pixels → field meters. Pixel coordinates may be negative or exceed the image size (off-frame landmarks).

---

### Task 1: Dependencies

**Files:**
- Modify: `experiments/pyproject.toml` (via poetry)

- [ ] **Step 1: Add matplotlib + PyQt6**

matplotlib is already a transitive dep of ultralytics but needs an interactive GUI backend for the click tool; PyQt6 provides one without system packages.

```bash
cd /home/robin/personal/touch-rugby-analyzer/experiments
poetry add matplotlib pyqt6
```

- [ ] **Step 2: Verify the interactive backend loads**

```bash
poetry run python -c "import matplotlib; matplotlib.use('QtAgg'); import matplotlib.pyplot; print('QtAgg ok')"
```

Expected: `QtAgg ok`

---

### Task 2: Shared geometry module

**Files:**
- Create: `experiments/field_calib.py`

- [ ] **Step 1: Write `field_calib.py`**

```python
"""Field model + homography helpers shared by calibrate_field.py / project_overhead.py.

Coordinate system (meters): X = 0-50 across the field (left sideline -> right,
as seen from the camera), Y = 0 at the camera's score line -> 70 at the far one.
Pixel coordinates may lie outside the image bounds: off-frame landmarks are
clicked on a zoomed-out canvas, so negative / > image-size values are expected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FIELD_LENGTH_M = 70.0  # score line to score line (Y axis)
FIELD_WIDTH_M = 50.0   # sideline to sideline (X axis)
MARGIN_M = 5.0         # tolerance beyond the lines before a point is "off field"
PITCH_MARGIN_PX = 30

# (name, X, Y) in meters, clicked in this order by calibrate_field.py
LANDMARKS: list[tuple[str, float, float]] = [
    ("near-left corner", 0.0, 0.0),
    ("near-right corner", FIELD_WIDTH_M, 0.0),
    ("halfway-left", 0.0, FIELD_LENGTH_M / 2),
    ("halfway-right", FIELD_WIDTH_M, FIELD_LENGTH_M / 2),
    ("far-left corner", 0.0, FIELD_LENGTH_M),
    ("far-right corner", FIELD_WIDTH_M, FIELD_LENGTH_M),
]


@dataclass
class Calibration:
    video: str
    image_size: tuple[int, int]  # (width, height)
    points: list[dict]           # {name, px, py, X, Y}
    homography: np.ndarray       # 3x3, image px -> field meters


def calibration_path(video: Path) -> Path:
    return video.with_suffix(".calibration.json")


def fit_homography(points: list[dict]) -> np.ndarray:
    if len(points) < 4:
        raise ValueError(f"Need >= 4 points, got {len(points)}")
    src = np.array([[p["px"], p["py"]] for p in points], dtype=np.float64)
    dst = np.array([[p["X"], p["Y"]] for p in points], dtype=np.float64)
    method = cv2.RANSAC if len(points) > 4 else 0
    H, _ = cv2.findHomography(src, dst, method=method, ransacReprojThreshold=1.0)
    if H is None:
        raise ValueError("Homography fit failed - points may be collinear")
    return H


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to an (N, 2) array of points, returning (N, 2)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def reprojection_errors_m(H: np.ndarray, points: list[dict]) -> np.ndarray:
    src = np.array([[p["px"], p["py"]] for p in points], dtype=np.float64)
    dst = np.array([[p["X"], p["Y"]] for p in points], dtype=np.float64)
    return np.linalg.norm(project(H, src) - dst, axis=1)


def in_field(field_pts: np.ndarray, margin: float = MARGIN_M) -> np.ndarray:
    """Boolean mask of (N, 2) field-meter points within the field + margin."""
    x, y = field_pts[:, 0], field_pts[:, 1]
    return (
        (x >= -margin) & (x <= FIELD_WIDTH_M + margin)
        & (y >= -margin) & (y <= FIELD_LENGTH_M + margin)
    )


def save(calib: Calibration, path: Path) -> None:
    data = {
        "video": calib.video,
        "image_size": list(calib.image_size),
        "field": {"length": FIELD_LENGTH_M, "width": FIELD_WIDTH_M},
        "points": calib.points,
        "homography": calib.homography.tolist(),
    }
    path.write_text(json.dumps(data, indent=2))


def load(path: Path) -> Calibration:
    data = json.loads(path.read_text())
    return Calibration(
        video=data["video"],
        image_size=tuple(data["image_size"]),
        points=data["points"],
        homography=np.array(data["homography"], dtype=np.float64),
    )


def draw_grid_overlay(frame: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project a 10 m field grid onto the camera frame for visual checking.

    Field lines are straight under a homography, but we sample each line
    densely and draw segment-by-segment so parts that land far outside the
    frame (or behind the horizon, |w| ~ 0) are simply dropped.
    """
    out = frame.copy()
    h, w = frame.shape[:2]
    H_inv = np.linalg.inv(H)

    def draw_field_line(p0, p1, color, thickness):
        ts = np.linspace(0.0, 1.0, 200)
        field_pts = np.outer(1 - ts, p0) + np.outer(ts, p1)  # (200, 2)
        ones = np.ones((len(field_pts), 1))
        img_h = (H_inv @ np.hstack([field_pts, ones]).T).T
        ok = np.abs(img_h[:, 2]) > 1e-9
        img = np.full((len(field_pts), 2), np.inf)
        img[ok] = img_h[ok, :2] / img_h[ok, 2:3]
        ok &= (np.abs(img[:, 0]) < 4 * w) & (np.abs(img[:, 1]) < 4 * h)
        for i in range(1, len(img)):
            if ok[i - 1] and ok[i]:
                a = (int(round(img[i - 1, 0])), int(round(img[i - 1, 1])))
                b = (int(round(img[i, 0])), int(round(img[i, 1])))
                cv2.line(out, a, b, color, thickness, cv2.LINE_AA)

    cyan, orange = (255, 255, 0), (0, 140, 255)
    for x in np.arange(10.0, FIELD_WIDTH_M, 10.0):
        draw_field_line((x, 0.0), (x, FIELD_LENGTH_M), cyan, 1)
    for y in np.arange(10.0, FIELD_LENGTH_M, 10.0):
        draw_field_line((0.0, y), (FIELD_WIDTH_M, y), cyan, 1)
    # boundary (score lines + sidelines) in orange, thicker
    draw_field_line((0.0, 0.0), (FIELD_WIDTH_M, 0.0), orange, 2)
    draw_field_line((0.0, FIELD_LENGTH_M), (FIELD_WIDTH_M, FIELD_LENGTH_M), orange, 2)
    draw_field_line((0.0, 0.0), (0.0, FIELD_LENGTH_M), orange, 2)
    draw_field_line((FIELD_WIDTH_M, 0.0), (FIELD_WIDTH_M, FIELD_LENGTH_M), orange, 2)
    return out


def make_pitch_canvas(px_per_m: float = 9.0) -> tuple[np.ndarray, "callable"]:
    """Return (canvas, to_px): an empty minimap and a meters->pixels mapper.

    The camera's score line (Y=0) is at the BOTTOM of the minimap, matching
    the camera's point of view.
    """
    m = PITCH_MARGIN_PX
    cw = int(round(FIELD_WIDTH_M * px_per_m + 2 * m))
    ch = int(round(FIELD_LENGTH_M * px_per_m + 2 * m))
    canvas = np.full((ch, cw, 3), (30, 90, 30), dtype=np.uint8)  # green, BGR

    def to_px(x_m: float, y_m: float) -> tuple[int, int]:
        return (int(round(m + x_m * px_per_m)), int(round(ch - m - y_m * px_per_m)))

    white = (255, 255, 255)
    cv2.rectangle(canvas, to_px(0, FIELD_LENGTH_M), to_px(FIELD_WIDTH_M, 0), white, 2)
    cv2.line(canvas, to_px(0, FIELD_LENGTH_M / 2), to_px(FIELD_WIDTH_M, FIELD_LENGTH_M / 2), white, 1)
    return canvas, to_px


PITCH_BG_BGR = (30, 90, 30)
```

- [ ] **Step 2: Verify with a synthetic round-trip**

A synthetic camera (known homography) generates pixel positions for the six landmarks; the fitted H must recover field coordinates to ≲1e-6 m, and `in_field` must accept/reject correctly.

```bash
cd /home/robin/personal/touch-rugby-analyzer/experiments
poetry run python -c "
import numpy as np, field_calib as fc
H_true = np.array([[0.05, -0.02, -10.0], [0.001, 0.09, -3.0], [0.0, 0.001, 1.0]])
H_inv = np.linalg.inv(H_true)
pts = []
for name, X, Y in fc.LANDMARKS:
    v = H_inv @ np.array([X, Y, 1.0]); px, py = v[0]/v[2], v[1]/v[2]
    pts.append({'name': name, 'px': px, 'py': py, 'X': X, 'Y': Y})
H = fc.fit_homography(pts)
err = fc.reprojection_errors_m(H, pts)
assert err.max() < 1e-6, err
assert fc.in_field(np.array([[25.0, 35.0], [-4.0, 0.0]])).all()
assert not fc.in_field(np.array([[60.0, 35.0], [25.0, 80.0]])).any()
canvas, to_px = fc.make_pitch_canvas(9.0)
assert canvas.shape[0] > canvas.shape[1]  # field is longer than wide
print('field_calib ok, max reproj err:', err.max())
"
```

Expected: `field_calib ok, max reproj err: <tiny number>`

---

### Task 3: Calibration click tool

**Files:**
- Create: `experiments/calibrate_field.py`

- [ ] **Step 1: Write `calibrate_field.py`**

```python
"""Interactive one-time calibration of a fixed-camera clip.

Opens frame 0 on a canvas with 50% padding beyond the image on every side, so
landmarks outside the camera frame (e.g. a corner the camera cuts off) can be
clicked at their extrapolated positions - out-of-image pixel coordinates are
valid homography input. Walks through `field_calib.LANDMARKS` in order.

Controls (matplotlib window):
    left-click          place the current landmark (ignored while the toolbar
                        pan/zoom tool is active - deactivate it to click)
    right-click or u    undo the last action
    s                   skip the current landmark (not visible / unknown)
    enter               finish early (needs >= 4 placed points)

Outputs, next to the clip:
    <clip>.calibration.json        points + fitted homography (px -> meters)
    <clip>.calibration_check.png   frame 0 with a projected 10 m grid - LOOK AT
                                   THIS before trusting the calibration.

Usage:
    poetry run python calibrate_field.py try_example.mp4
    poetry run python calibrate_field.py try_example.mp4 --refit   # headless:
        refit H + regenerate the check image from the existing JSON's points
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

import field_calib as fc

log = logging.getLogger("calibrate_field")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path, help="Input MP4")
    p.add_argument("--refit", action="store_true",
                   help="No GUI: refit homography from the existing calibration JSON")
    return p.parse_args()


def read_frame0(video: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {video}")
    return frame


class ClickUI:
    """Walks the user through LANDMARKS; collects {name, px, py, X, Y} dicts."""

    def __init__(self, frame_bgr: np.ndarray):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.points: list[dict] = []
        self.history: list[str] = []  # "placed" | "skipped", for undo
        self.artists: list = []
        self.idx = 0
        h, w = frame_bgr.shape[:2]
        self.fig, self.ax = plt.subplots(figsize=(15, 9))
        self.ax.imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), extent=(0, w, h, 0))
        self.ax.set_xlim(-0.5 * w, 1.5 * w)
        self.ax.set_ylim(1.5 * h, -0.5 * h)
        self.ax.set_facecolor("#15171f")
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self._refresh_title()

    def _refresh_title(self) -> None:
        if self.idx < len(fc.LANDMARKS):
            name, X, Y = fc.LANDMARKS[self.idx]
            t = (f"[{self.idx + 1}/{len(fc.LANDMARKS)}] Click: {name}  (X={X:g}m, Y={Y:g}m)"
                 f"   |   s=skip  u/right-click=undo  enter=finish")
        else:
            t = "All landmarks done - close the window (or press enter)"
        self.ax.set_title(t, fontsize=11)
        self.fig.canvas.draw_idle()

    def on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self.ax.get_navigate_mode():  # pan/zoom tool active
            return
        if event.button == 1 and self.idx < len(fc.LANDMARKS):
            name, X, Y = fc.LANDMARKS[self.idx]
            self.points.append(
                {"name": name, "px": float(event.xdata), "py": float(event.ydata), "X": X, "Y": Y}
            )
            self.history.append("placed")
            (dot,) = self.ax.plot(event.xdata, event.ydata, "o", ms=8, mec="white", mfc="red")
            label = self.ax.annotate(name, (event.xdata, event.ydata),
                                     xytext=(8, -8), textcoords="offset points",
                                     color="white", fontsize=9)
            self.artists.append((dot, label))
            self.idx += 1
            self._refresh_title()
        elif event.button == 3:
            self.undo()

    def on_key(self, event) -> None:
        if event.key == "s" and self.idx < len(fc.LANDMARKS):
            self.history.append("skipped")
            self.idx += 1
            self._refresh_title()
        elif event.key == "u":
            self.undo()
        elif event.key == "enter":
            self.plt.close(self.fig)

    def undo(self) -> None:
        if not self.history:
            return
        if self.history.pop() == "placed":
            self.points.pop()
            for artist in self.artists.pop():
                artist.remove()
        self.idx -= 1
        self._refresh_title()

    def run(self) -> list[dict]:
        self.plt.show()  # blocks until the window is closed
        return self.points


def fit_report_save(video: Path, frame: np.ndarray, points: list[dict]) -> None:
    if len(points) < 4:
        raise SystemExit(f"Got {len(points)} points - need at least 4. Nothing saved.")
    H = fc.fit_homography(points)
    errors = fc.reprojection_errors_m(H, points)
    log.info("Reprojection error per landmark:")
    for p, e in zip(points, errors):
        log.info("  %-18s (%.0f, %.0f)px -> err %.2f m", p["name"], p["px"], p["py"], e)
    log.info("Mean error: %.2f m", errors.mean())
    if errors.mean() > 1.0:
        log.warning("Mean reprojection error > 1 m - re-click the worst landmarks"
                    " (lens distortion or a misplaced point).")

    h, w = frame.shape[:2]
    calib = fc.Calibration(video=video.name, image_size=(w, h), points=points, homography=H)
    json_path = fc.calibration_path(video)
    fc.save(calib, json_path)

    check = fc.draw_grid_overlay(frame, H)
    for p in points:
        pt = (int(round(p["px"])), int(round(p["py"])))
        cv2.drawMarker(check, pt, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(check, p["name"], (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    check_path = video.with_suffix(".calibration_check.png")
    cv2.imwrite(str(check_path), check)
    log.info("Saved %s and %s", json_path.name, check_path.name)
    log.info("Open %s and check the grid hugs the cones before projecting.", check_path.name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
    args = parse_args()
    if not args.video.exists():
        raise SystemExit(f"No file at {args.video}")
    frame = read_frame0(args.video)

    if args.refit:
        json_path = fc.calibration_path(args.video)
        if not json_path.exists():
            raise SystemExit(f"--refit needs an existing {json_path}")
        points = fc.load(json_path).points
    else:
        points = ClickUI(frame).run()
    fit_report_save(args.video, frame, points)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Headless verification via `--refit`**

Build a synthetic calibration JSON (same synthetic camera as Task 2), then run the refit path end-to-end against the real clip frame:

```bash
cd /home/robin/personal/touch-rugby-analyzer/experiments
poetry run python -c "
import json, numpy as np, field_calib as fc
H_true = np.array([[0.05, -0.02, -10.0], [0.001, 0.09, -3.0], [0.0, 0.001, 1.0]])
H_inv = np.linalg.inv(H_true)
pts = []
for name, X, Y in fc.LANDMARKS:
    v = H_inv @ np.array([X, Y, 1.0])
    pts.append({'name': name, 'px': v[0]/v[2], 'py': v[1]/v[2], 'X': X, 'Y': Y})
fc.save(fc.Calibration('try_example.mp4', (1280, 720), pts, H_true),
        fc.calibration_path(__import__('pathlib').Path('try_example.mp4')))
print('synthetic calibration written')
"
poetry run python calibrate_field.py try_example.mp4 --refit
ls -la try_example.calibration.json try_example.calibration_check.png
```

Expected: per-landmark errors ~0.00 m, both files exist. (The check PNG's grid will look wrong on the real frame — the synthetic camera doesn't match the real one. That's expected; it verifies the code path, not the geometry.)

- [ ] **Step 3: GUI smoke test (quick)**

Launch the window, click 2–3 arbitrary points, press `u` (undo), `s` (skip), then close without enough points:

```bash
poetry run python calibrate_field.py try_example.mp4
```

Expected: window opens zoomed out with dark padding around the frame; clicks register **outside the image area too**; closing with < 4 points exits with `Got N points - need at least 4. Nothing saved.` (If running where no display is available, defer this to the final user calibration run.)

---

### Task 4: Overhead projection pipeline

**Files:**
- Create: `experiments/project_overhead.py`

- [ ] **Step 1: Write `project_overhead.py`**

```python
"""Track players and project them onto an overhead minimap (fixed camera).

Requires a calibration produced by calibrate_field.py. Reuses the YOLO11 +
BoT-SORT setup from track_humans.py: detections filtered to `person`, per-track
EMA smoothing of the bottom-center anchor (in pixel space) before projection.

Outputs, next to the clip:
    <clip>_overhead.mp4   side-by-side: camera view (boxes/IDs) | minimap (dots + trails)
    <clip>_tracks.csv     frame, time_s, track_id, x_m, y_m
    <clip>_trails.png     all full trails on one pitch diagram

Usage:
    poetry run python project_overhead.py try_example.mp4
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

import field_calib as fc

PERSON_CLASS_ID = 0  # COCO

log = logging.getLogger("project_overhead")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path, help="Input MP4")
    p.add_argument("--calibration", type=Path, default=None,
                   help="Calibration JSON (default: <video>.calibration.json)")
    p.add_argument("--output", type=Path, default=None, help="Output MP4 (default: <input>_overhead.mp4)")
    p.add_argument("--model", default="yolo11m.pt", help="Ultralytics model name or path")
    p.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=1280, help="Inference image size (long side)")
    p.add_argument("--smoothing", type=float, default=0.6,
                   help="EMA weight of the previous anchor position (0=raw, ~0.9=heavy)")
    p.add_argument("--trail-seconds", type=float, default=3.0,
                   help="Length of the fading minimap trail, in seconds")
    p.add_argument("--device", default=None, help="cuda device, e.g. '0' or 'cpu' (default: auto)")
    p.add_argument("--log-every", type=int, default=30, help="Frames between progress lines")
    return p.parse_args()


def probe_video(path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, w, h, n


def track_color(tid: int) -> tuple[int, int, int]:
    return sv.ColorPalette.DEFAULT.by_idx(tid).as_bgr()


def smooth_anchors(
    detections: sv.Detections, state: dict[int, np.ndarray], alpha: float
) -> np.ndarray:
    """Per-track EMA of the bottom-center anchor, in pixel space.

    Same scheme as track_humans._smoothed_for_trace, but returns the (N, 2)
    anchors directly instead of shifting boxes.
    """
    xyxy = detections.xyxy.astype(np.float32)
    raw = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2.0, xyxy[:, 3]], axis=1)
    out = raw.copy()
    for i, tid in enumerate(detections.tracker_id):
        tid = int(tid)
        prev = state.get(tid)
        out[i] = raw[i] if prev is None or alpha <= 0 else alpha * prev + (1.0 - alpha) * raw[i]
        state[tid] = out[i]
    return out


def draw_trail(canvas: np.ndarray, pts: list[tuple[int, int]],
               color: tuple[int, int, int], max_age: int) -> None:
    """Polyline whose older segments fade toward the pitch background."""
    n = len(pts)
    for i in range(max(1, n - max_age), n):
        f = 1.0 - (n - 1 - i) / max(1, max_age)
        c = tuple(int(b + (col - b) * f) for b, col in zip(fc.PITCH_BG_BGR, color))
        cv2.line(canvas, pts[i - 1], pts[i], c, 2, cv2.LINE_AA)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    if not args.video.exists():
        raise SystemExit(f"No file at {args.video}")
    calib_path = args.calibration or fc.calibration_path(args.video)
    if not calib_path.exists():
        raise SystemExit(
            f"No calibration at {calib_path} - run: poetry run python calibrate_field.py {args.video.name}"
        )
    calib = fc.load(calib_path)
    H = calib.homography

    fps, w, h, total_frames = probe_video(args.video)
    log.info("Video: %dx%d @ %.2f fps | %d frames", w, h, fps, total_frames)
    if (w, h) != calib.image_size:
        log.warning("Calibration was made for %sx%s but video is %dx%d",
                    *calib.image_size, w, h)

    # Minimap sized to match the video height exactly.
    px_per_m = (h - 2 * fc.PITCH_MARGIN_PX) / fc.FIELD_LENGTH_M
    pitch, to_px = fc.make_pitch_canvas(px_per_m)
    if pitch.shape[0] > h:      # guard against 1px rounding drift
        pitch = pitch[:h]
    elif pitch.shape[0] < h:
        pad = np.full((h - pitch.shape[0], pitch.shape[1], 3), fc.PITCH_BG_BGR, dtype=np.uint8)
        pitch = np.vstack([pitch, pad])
    map_w = pitch.shape[1]

    out_path = args.output or args.video.with_name(f"{args.video.stem}_overhead.mp4")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w + map_w, h))

    palette = sv.ColorPalette.DEFAULT
    box_annotator = sv.BoxAnnotator(color=palette, thickness=2, color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(color=palette, text_scale=0.4, text_thickness=1,
                                        color_lookup=sv.ColorLookup.TRACK)

    model = YOLO(args.model)
    stream = model.track(
        source=str(args.video), stream=True, persist=True, tracker="botsort.yaml",
        classes=[PERSON_CLASS_ID], conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        device=args.device, verbose=False,
    )

    trail_frames = max(1, int(round(args.trail_seconds * fps)))
    anchor_state: dict[int, np.ndarray] = {}
    # full minimap-pixel history per track (trails PNG + fading video trail)
    history_px: dict[int, list[tuple[int, int]]] = defaultdict(list)
    csv_rows: list[tuple[int, float, int, float, float]] = []
    dropped = 0
    frame_idx = 0
    t0 = time.perf_counter()

    for result in stream:
        frame = result.orig_img.copy()
        detections = sv.Detections.from_ultralytics(result)
        if detections.tracker_id is not None:
            keep = detections.tracker_id != -1
            detections = detections[keep] if keep.any() else detections[np.zeros(len(detections), dtype=bool)]
        else:
            detections = detections[np.zeros(len(detections), dtype=bool)]

        if len(detections):
            anchors = smooth_anchors(detections, anchor_state, args.smoothing)
            field_pts = fc.project(H, anchors)
            mask = fc.in_field(field_pts)
            dropped += int((~mask).sum())
            detections = detections[mask]
            field_pts = field_pts[mask]
        else:
            field_pts = np.empty((0, 2))

        tids = detections.tracker_id if detections.tracker_id is not None else []
        for tid, (x_m, y_m) in zip(tids, field_pts):
            tid = int(tid)
            history_px[tid].append(to_px(x_m, y_m))
            csv_rows.append((frame_idx, frame_idx / fps, tid, round(float(x_m), 2), round(float(y_m), 2)))

        if len(detections):
            labels = [f"#{tid}" for tid in detections.tracker_id]
            frame = box_annotator.annotate(frame, detections)
            frame = label_annotator.annotate(frame, detections, labels=labels)

        minimap = pitch.copy()
        live = set(int(t) for t in (detections.tracker_id if len(detections) else []))
        for tid, pts in history_px.items():
            draw_trail(minimap, pts, track_color(tid), trail_frames)
            if tid in live:
                cv2.circle(minimap, pts[-1], 6, track_color(tid), -1, cv2.LINE_AA)
                cv2.circle(minimap, pts[-1], 6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(minimap, str(tid), (pts[-1][0] + 8, pts[-1][1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(np.hstack([frame, minimap]))
        frame_idx += 1
        if frame_idx % args.log_every == 0:
            fps_avg = frame_idx / (time.perf_counter() - t0)
            log.info("[%4d/%d] %.1f fps | %d tracks on field | %d dropped off-field",
                     frame_idx, total_frames, fps_avg, len(live), dropped)

    writer.release()

    csv_path = args.video.with_name(f"{args.video.stem}_tracks.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s", "track_id", "x_m", "y_m"])
        wr.writerows(csv_rows)

    trails = pitch.copy()
    for tid, pts in history_px.items():
        if len(pts) > 1:
            cv2.polylines(trails, [np.array(pts, dtype=np.int32)], False, track_color(tid), 2, cv2.LINE_AA)
        cv2.putText(trails, str(tid), (pts[-1][0] + 6, pts[-1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    png_path = args.video.with_name(f"{args.video.stem}_trails.png")
    cv2.imwrite(str(png_path), trails)

    log.info("=" * 60)
    log.info("Done: %d frames | %d tracks | %d positions | %d dropped off-field",
             frame_idx, len(history_px), len(csv_rows), dropped)
    log.info("  %s", out_path.resolve())
    log.info("  %s", csv_path.resolve())
    log.info("  %s", png_path.resolve())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: End-to-end run with a placeholder calibration**

The synthetic calibration from Task 3 Step 2 doesn't match the real camera, so first write a **rough eyeballed** calibration for `try_example.mp4` (estimated cone pixel positions read off the extracted frame — good enough to verify the pipeline; the user's clicked calibration replaces it). The executor should read pixel coordinates of the near orange cones / mid white cones from `/tmp/try_frame0.png` crops and write the JSON via a short script using `field_calib.save`, then:

```bash
cd /home/robin/personal/touch-rugby-analyzer/experiments
poetry run python project_overhead.py try_example.mp4
ls -la try_example_overhead.mp4 try_example_tracks.csv try_example_trails.png
head -5 try_example_tracks.csv
```

Expected: all three outputs exist; CSV has header + rows with x_m in roughly [-5, 55] and y_m in [-5, 75]; log reports >0 tracks and some dropped off-field points (spectators).

- [ ] **Step 3: Visual check of the output video**

```bash
ffmpeg -y -v error -i try_example_overhead.mp4 -vf "select=eq(n\,150)" -vframes 1 -vsync vfr /tmp/overhead_check.png
```

Read `/tmp/overhead_check.png` and confirm: left pane shows boxes/IDs on players; right pane shows the pitch outline with colored dots in plausible positions (cluster of players mid-field, nobody pinned to a corner); trails fade.

---

### Task 5: User calibration handoff

- [ ] **Step 1: User runs the real calibration**

```bash
cd /home/robin/personal/touch-rugby-analyzer/experiments
poetry run python calibrate_field.py try_example.mp4   # click the cones
# then check try_example.calibration_check.png — grid must hug the cones
poetry run python project_overhead.py try_example.mp4
```

- [ ] **Step 2: Review results together**

Check straight-line runs produce straight trails; decide whether phase-2 (k1 undistortion) is needed.
