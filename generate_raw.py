from pathlib import Path
import hashlib
import struct
import zlib

import numpy as np
import pandas as pd


QUESTION_TYPES = [
    "best_quadrant",
    "contamination_quadrant",
    "usable_hole_count",
    "crack_orientation",
    "ice_thickness",
    "fiducial_count",
]

CHOICES = {
    "best_quadrant": ["upper left", "upper right", "lower left", "lower right"],
    "contamination_quadrant": ["upper left", "upper right", "lower left", "lower right"],
    "usable_hole_count": ["0-3", "4-6", "7-9", "10 or more"],
    "crack_orientation": ["no dominant crack", "horizontal", "vertical", "diagonal"],
    "ice_thickness": ["thin", "medium", "thick", "mixed"],
    "fiducial_count": ["0", "1", "2", "3 or more"],
}


def _write_png(path: Path, arr: np.ndarray) -> None:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def _disk(arr, cx, cy, r, color, alpha=1.0):
    yy, xx = np.ogrid[: arr.shape[0], : arr.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    arr[mask] = (1 - alpha) * arr[mask] + alpha * np.array(color)


def _ellipse(arr, cx, cy, rx, ry, color, alpha=1.0):
    yy, xx = np.ogrid[: arr.shape[0], : arr.shape[1]]
    mask = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1.0
    arr[mask] = (1 - alpha) * arr[mask] + alpha * np.array(color)


def _line(arr, p0, p1, width, color, alpha=1.0):
    x0, y0 = p0
    x1, y1 = p1
    steps = int(max(abs(x1 - x0), abs(y1 - y0), 1)) + 1
    for t in np.linspace(0, 1, steps):
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        _disk(arr, x, y, max(1, width // 2), color, alpha)


def _rect(arr, x0, y0, x1, y1, color, alpha=1.0):
    x0, x1 = sorted((max(0, x0), min(arr.shape[1], x1)))
    y0, y1 = sorted((max(0, y0), min(arr.shape[0], y1)))
    arr[y0:y1, x0:x1] = (1 - alpha) * arr[y0:y1, x0:x1] + alpha * np.array(color)


def _quadrant(cx, cy):
    if cy < 80 and cx < 80:
        return "upper_left"
    if cy < 80 and cx >= 80:
        return "upper_right"
    if cy >= 80 and cx < 80:
        return "lower_left"
    return "lower_right"


def _answer_quadrant(q):
    return {"upper_left": "A", "upper_right": "B", "lower_left": "C", "lower_right": "D"}[q]


def _blur3(arr):
    padded = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="edge")
    return (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0


def _draw_hole(arr, rng, x, y, state, visibility):
    rim = [82, 87, 88]
    rx = int(rng.integers(11, 15))
    ry = int(rng.integers(10, 14))
    _ellipse(arr, x, y, rx, ry, rim, 0.72)
    for _ in range(4):
        _disk(
            arr,
            x + int(rng.normal(0, 4)),
            y + int(rng.normal(0, 4)),
            int(rng.integers(2, 5)),
            [65, 70, 72],
            0.25,
        )
    base = {
        "thin": [45, 51, 52],
        "medium": [72, 78, 79],
        "thick": [118, 123, 121],
        "broken": [32, 35, 36],
        "contaminated": [94, 90, 86],
    }[state]
    _ellipse(arr, x + int(rng.normal(0, 1)), y + int(rng.normal(0, 1)), rx - 4, ry - 4, base, 0.86)
    for _ in range(10):
        px = x + int(rng.normal(0, max(2, rx / 2.8)))
        py = y + int(rng.normal(0, max(2, ry / 2.8)))
        _disk(arr, px, py, 1, np.array(base) + rng.normal(0, 16, 3), 0.20)
    if state == "thick":
        for _ in range(5):
            _disk(arr, x + int(rng.normal(0, 5)), y + int(rng.normal(0, 5)), int(rng.integers(1, 3)), [155, 160, 158], 0.45)
    elif state == "broken":
        _line(arr, (x - 7, y - 4), (x + 7, y + 5), 2, [18, 20, 21], 0.65)
        _line(arr, (x - 3, y + 7), (x + 5, y - 7), 1, [20, 22, 23], 0.60)
    elif state == "contaminated":
        for _ in range(10):
            col = [24, 24, 23] if rng.random() < 0.62 else [150, 146, 138]
            _disk(arr, x + int(rng.normal(0, 5)), y + int(rng.normal(0, 5)), int(rng.integers(1, 3)), col, 0.8)
    if visibility == "low_dose":
        _disk(arr, x, y, 10, [92, 96, 96], 0.18)


def _add_global_artifacts(arr, rng, visibility, ood_axis):
    if visibility == "frosty":
        for _ in range(65):
            _disk(arr, int(rng.integers(0, 160)), int(rng.integers(0, 160)), int(rng.choice([1, 1, 2])), [178, 181, 177], 0.30)
    elif visibility == "low_dose":
        arr[:] = 128 + (arr - 128) * 0.62
    elif visibility == "crowded":
        for _ in range(18):
            _disk(arr, int(rng.integers(0, 160)), int(rng.integers(0, 160)), int(rng.integers(2, 6)), [62, 64, 62], 0.42)

    if ood_axis == "carbon_shadow":
        _rect(arr, 0, int(rng.integers(0, 55)), 160, int(rng.integers(35, 90)), [22, 24, 25], 0.34)
    elif ood_axis == "beam_gradient":
        gradient = np.linspace(0.72, 1.16, 160)[None, :, None]
        arr[:] = arr * gradient
    elif ood_axis == "ice_rings":
        yy, xx = np.ogrid[:160, :160]
        for r in [31, 52, 71]:
            band = np.abs(np.sqrt((xx - 80) ** 2 + (yy - 80) ** 2) - r) <= 1.4
            arr[band] = 0.82 * arr[band] + 0.18 * np.array([152, 156, 154])

    arr[:] = arr + rng.normal(0, {"clear": 4, "frosty": 7, "low_dose": 9, "crowded": 8}[visibility], arr.shape)


def _scene_trace(rng):
    difficulty = rng.choice(["easy", "medium", "hard"], p=[0.28, 0.47, 0.25])
    visibility = (
        rng.choice(["clear", "frosty"], p=[0.80, 0.20])
        if difficulty == "easy"
        else rng.choice(["clear", "frosty", "low_dose", "crowded"], p=[0.30, 0.27, 0.23, 0.20])
        if difficulty == "medium"
        else rng.choice(["frosty", "low_dose", "crowded"], p=[0.33, 0.34, 0.33])
    )
    ood_axis = rng.choice(
        ["standard", "carbon_shadow", "beam_gradient", "ice_rings"],
        p=[0.62, 0.14, 0.12, 0.12] if difficulty != "hard" else [0.26, 0.28, 0.23, 0.23],
    )
    trace = {
        "difficulty": difficulty,
        "visibility": visibility,
        "layout_family": rng.choice(["square", "offset", "compressed"], p=[0.50, 0.28, 0.22]),
        "ood_axis": ood_axis,
        "contamination_quadrant": rng.choice(["upper_left", "upper_right", "lower_left", "lower_right"]),
        "crack_orientation": rng.choice(["none", "horizontal", "vertical", "diagonal"], p=[0.32, 0.22, 0.22, 0.24]),
        "fiducial_count": int(rng.choice([0, 1, 2, 3, 4], p=[0.18, 0.27, 0.25, 0.20, 0.10])),
        "dominant_ice": rng.choice(["thin", "medium", "thick", "mixed"], p=[0.25, 0.32, 0.23, 0.20]),
    }
    return trace


def _draw_scene(rng, trace):
    arr = np.zeros((160, 160, 3), dtype=float)
    yy, xx = np.mgrid[:160, :160]
    base = 72 + 5.5 * np.sin(xx / 17.0 + rng.random()) + 4.0 * np.cos(yy / 23.0)
    vignette = 1.0 - 0.22 * (((xx - 80) ** 2 + (yy - 80) ** 2) / (80**2))
    grayscale = base * vignette + rng.normal(0, 3.6, (160, 160))
    arr[:] = grayscale[:, :, None]
    for x in [0, int(rng.integers(145, 160))]:
        _rect(arr, x, 0, min(160, x + int(rng.integers(9, 18))), 160, [25, 27, 28], 0.28)
    for y in [0, int(rng.integers(145, 160))]:
        _rect(arr, 0, y, 160, min(160, y + int(rng.integers(9, 18))), [27, 29, 30], 0.20)

    if trace["layout_family"] == "compressed":
        xs = [42, 67, 92, 117]
        ys = [38, 66, 94, 122]
    elif trace["layout_family"] == "offset":
        xs = [38, 64, 92, 122]
        ys = [36, 64, 94, 124]
    else:
        xs = [38, 66, 94, 122]
        ys = [38, 66, 94, 122]

    state_counts = {"thin": 0, "medium": 0, "thick": 0, "broken": 0, "contaminated": 0}
    quadrant_usable = {"upper_left": 0, "upper_right": 0, "lower_left": 0, "lower_right": 0}
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            if trace["layout_family"] == "offset" and row % 2:
                x += 5
            q = _quadrant(x, y)
            if q == trace["contamination_quadrant"] and rng.random() < (0.45 if trace["difficulty"] != "easy" else 0.32):
                state = "contaminated"
            elif rng.random() < (0.10 if trace["difficulty"] == "easy" else 0.17):
                state = "broken"
            elif trace["dominant_ice"] == "mixed":
                state = rng.choice(["thin", "medium", "thick"], p=[0.30, 0.40, 0.30])
            else:
                state = trace["dominant_ice"]
            state_counts[state] += 1
            if state in {"thin", "medium"}:
                quadrant_usable[q] += 1
            _draw_hole(arr, rng, x, y, state, trace["visibility"])

    trace["usable_holes"] = quadrant_usable["upper_left"] + quadrant_usable["upper_right"] + quadrant_usable["lower_left"] + quadrant_usable["lower_right"]
    trace["best_quadrant"] = max(sorted(quadrant_usable), key=lambda q: quadrant_usable[q])

    if trace["crack_orientation"] == "horizontal":
        _line(arr, (8, int(rng.integers(72, 89))), (152, int(rng.integers(72, 89))), 3, [18, 19, 20], 0.62)
    elif trace["crack_orientation"] == "vertical":
        _line(arr, (int(rng.integers(72, 89)), 8), (int(rng.integers(72, 89)), 152), 3, [18, 19, 20], 0.62)
    elif trace["crack_orientation"] == "diagonal":
        _line(arr, (10, 16), (150, 144), 3, [18, 19, 20], 0.62)

    for _ in range(trace["fiducial_count"]):
        fx = int(rng.integers(20, 142))
        fy = int(rng.integers(20, 142))
        _disk(arr, fx, fy, int(rng.integers(2, 4)), [18, 18, 17], 0.85)
        _disk(arr, fx - 1, fy - 1, 1, [145, 146, 141], 0.30)

    _add_global_artifacts(arr, rng, trace["visibility"], trace["ood_axis"])
    arr[:] = 0.65 * arr + 0.35 * _blur3(arr)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _answer(trace, qtype):
    if qtype == "best_quadrant":
        return _answer_quadrant(trace["best_quadrant"])
    if qtype == "contamination_quadrant":
        return _answer_quadrant(trace["contamination_quadrant"])
    if qtype == "usable_hole_count":
        return "A" if trace["usable_holes"] <= 3 else "B" if trace["usable_holes"] <= 6 else "C" if trace["usable_holes"] <= 9 else "D"
    if qtype == "crack_orientation":
        return {"none": "A", "horizontal": "B", "vertical": "C", "diagonal": "D"}[trace["crack_orientation"]]
    if qtype == "ice_thickness":
        return {"thin": "A", "medium": "B", "thick": "C", "mixed": "D"}[trace["dominant_ice"]]
    if qtype == "fiducial_count":
        return "A" if trace["fiducial_count"] == 0 else "B" if trace["fiducial_count"] == 1 else "C" if trace["fiducial_count"] == 2 else "D"
    raise ValueError(qtype)


def _question(qtype):
    return {
        "best_quadrant": "Which quadrant contains the most usable thin-or-medium ice holes?",
        "contamination_quadrant": "Which quadrant contains the strongest contamination cluster?",
        "usable_hole_count": "How many usable holes are visible in the grid square?",
        "crack_orientation": "What is the dominant crack orientation?",
        "ice_thickness": "What is the dominant ice thickness condition?",
        "fiducial_count": "How many high-contrast fiducial beads are visible?",
    }[qtype]


def _stable_id(prefix, value):
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:14]}"


def main():
    rng = np.random.default_rng(90217)
    root = Path(__file__).resolve().parent
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for old in image_dir.glob("*.png"):
        old.unlink()

    rows = []
    scene_count = 820
    for i in range(scene_count):
        scene_id = f"raw_grid_{i:05d}"
        trace = _scene_trace(rng)
        image_path = f"images/{scene_id}.png"
        _write_png(root / image_path, _draw_scene(rng, trace))
        qtypes = list(rng.choice(QUESTION_TYPES, size=3, replace=False))
        if trace["difficulty"] == "hard" and "best_quadrant" not in qtypes:
            qtypes[0] = "best_quadrant"
        for qtype in qtypes:
            choices = CHOICES[qtype]
            rows.append(
                {
                    "question_id": _stable_id("raw_q", f"{scene_id}:{qtype}"),
                    "scene_id": scene_id,
                    "image_path": image_path,
                    "question_type": qtype,
                    "question": _question(qtype),
                    "choice_a": choices[0],
                    "choice_b": choices[1],
                    "choice_c": choices[2],
                    "choice_d": choices[3],
                    "answer_label": _answer(trace, qtype),
                    "difficulty": trace["difficulty"],
                    "visibility": trace["visibility"],
                    "layout_family": trace["layout_family"],
                    "ood_axis": trace["ood_axis"],
                    "trace_best_quadrant": trace["best_quadrant"],
                    "trace_contamination_quadrant": trace["contamination_quadrant"],
                    "trace_usable_holes": trace["usable_holes"],
                    "trace_crack_orientation": trace["crack_orientation"],
                    "trace_dominant_ice": trace["dominant_ice"],
                    "trace_fiducial_count": trace["fiducial_count"],
                }
            )
    pd.DataFrame(rows).to_csv(root / "data.csv", index=False)
    print(f"wrote {scene_count} images and {len(rows)} question rows")


if __name__ == "__main__":
    main()
