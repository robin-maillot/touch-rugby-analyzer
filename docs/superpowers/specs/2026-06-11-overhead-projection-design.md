# Fixed-camera overhead projection of player tracks

**Date:** 2026-06-11
**Status:** approved
**Scope:** `experiments/` (gitignored prototype area)

## Goal

From a short fixed-camera touch rugby clip (`experiments/try_example.mp4`, 1280×720
@30fps, camera on the score line), produce an "overhead" (bird's-eye) view of player
movement: an animated 2D minimap video, per-frame field coordinates as CSV, and a
static trails image.

## Approach

Fixed camera ⇒ a single homography per clip maps image pixels to field meters. Point
correspondences are captured once per clip with a manual click tool; player positions
come from the existing YOLO11 + BoT-SORT pipeline (`experiments/track_humans.py`).
No ML field-registration model needed for the fixed-camera case.

**Field coordinate system:** x = 0–50 m across the field (left sideline → right,
as seen from the camera), y = 0 at the camera's score line → 70 m at the far score
line. Field model: full FIT touch field, 70×50 m, cone-marked.

## Components

### 1. `experiments/calibrate_field.py <clip.mp4>`

One-time per clip. Interactive matplotlib tool:

- Loads frame 0, displays it with generous padding around the image (axes extend
  well beyond the image bounds), with pan/zoom enabled.
- **Clicks anywhere in the canvas are valid, including outside the image** — when a
  field corner is out of frame, the user zooms out and clicks its extrapolated
  position. Out-of-image pixel coordinates (negative, or > width/height) are stored
  as-is; `cv2.findHomography` accepts them.
- Prompts through a fixed landmark sequence (title bar shows current landmark):
  near-left corner (0,0), near-right corner (50,0), halfway-left (0,35),
  halfway-right (50,35), far-left corner (0,70), far-right corner (50,70).
  Keys: left-click = place point, right-click/`u` = undo, `s` = skip landmark,
  `enter` = finish early.
- Requires ≥ 4 non-collinear points; computes homography (`cv2.findHomography`,
  RANSAC when > 4 points), prints per-point reprojection error in meters.
- Saves `<clip>.calibration.json`:
  `{video, image_size, field: {length: 70, width: 50}, points: [{name, px, py, X, Y}], homography: [[...]]}`.
- Saves `<clip>.calibration_check.png`: frame 0 with a projected 10 m field grid
  overlaid, for visual verification that the grid hugs the cones.

### 2. `experiments/project_overhead.py <clip.mp4>`

Reads `<clip>.calibration.json` (clear error if missing). Pipeline:

- YOLO11m + BoT-SORT person tracking, same parameters as `track_humans.py`
  (conf 0.3, iou 0.5, imgsz 1280, person class only, `persist=True`).
- Per track per frame: EMA-smoothed bottom-center anchor (smoothing 0.6, as in
  `track_humans.py`) → perspective-transform through H → field meters.
- Points outside field + 5 m margin are dropped (filters spectators/passers-by).
- Outputs:
  - `<clip>_overhead.mp4` — side-by-side: left = original frame with boxes/IDs,
    right = 2D pitch minimap (field outline, score lines, halfway) with per-player
    colored dots and fading trails.
  - `<clip>_tracks.csv` — `frame, time_s, track_id, x_m, y_m`.
  - `<clip>_trails.png` — static plot of all full trails on the pitch.

## Lens distortion (deliberately deferred)

The clip has visible wide-angle distortion, strongest at frame edges. v1 ignores it;
the calibration grid overlay makes the resulting error visible. If trails are
noticeably bent, phase 2 adds a single-parameter (k1) radial undistortion applied to
both calibration clicks and track anchors before the homography.

## Error handling

- Calibrate: refuse to save with < 4 points or degenerate geometry; warn when mean
  reprojection error > 1 m.
- Project: hard error if calibration JSON missing or video unreadable; log count of
  dropped out-of-field points.

## Validation

No formal unit tests — `experiments/` is an exploratory, gitignored area (the repo's
`node test.js` suite covers the web app only). Verification is visual and built-in:

1. Calibration grid overlay must visually align with the cones/lines.
2. Reported reprojection error ≤ ~1 m mean.
3. In the output video, a player moving along a marked line should trace a straight
   trail at the corresponding field coordinate.