"""Small, local Haar-cascade face detector. No identity recognition or network I/O."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import cv2
import numpy as np

@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int

class FaceDetector:
    def __init__(self, *, scale_factor: float = 1.1, min_neighbors: int = 5,
                 min_size: int = 30, max_dimension: int = 1280) -> None:
        if not math.isfinite(scale_factor) or not 1.01 <= scale_factor <= 2.0:
            raise ValueError('scale_factor must be between 1.01 and 2.0.')
        if not 1 <= min_neighbors <= 50:
            raise ValueError('min_neighbors must be between 1 and 50.')
        if not 10 <= min_size <= 1000 or not 64 <= max_dimension <= 4096:
            raise ValueError('min_size must be 10..1000; max_dimension must be 64..4096.')
        self.scale_factor, self.min_neighbors = scale_factor, min_neighbors
        self.min_size, self.max_dimension = min_size, max_dimension
        model = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
        self.classifier = cv2.CascadeClassifier(str(model))
        if self.classifier.empty():
            raise RuntimeError('OpenCV could not load its bundled face cascade. Reinstall requirements.')

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError('Expected an unsigned 8-bit image array.')
        if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError('Expected a nonempty BGR image with three channels.')
        height, width = frame.shape[:2]
        if height * width > 40_000_000:
            raise ValueError('Image exceeds the 40 megapixel processing limit.')
        ratio = min(1.0, self.max_dimension / max(height, width))
        work = frame if ratio == 1.0 else cv2.resize(
            frame, (max(1, round(width * ratio)), max(1, round(height * ratio))),
            interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY))
        found = self.classifier.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size))
        sx, sy = width / work.shape[1], height / work.shape[0]
        boxes = []
        for x, y, w, h in found:
            x1, y1 = max(0, round(int(x)*sx)), max(0, round(int(y)*sy))
            x2, y2 = min(width, round(int(x+w)*sx)), min(height, round(int(y+h)*sy))
            if x2 > x1 and y2 > y1:
                boxes.append(FaceBox(x1, y1, x2-x1, y2-y1))
        return sorted(boxes, key=lambda b: (b.x, b.y, b.width, b.height))

def annotate(frame: np.ndarray, boxes: list[FaceBox], *, blur: bool = False) -> np.ndarray:
    """Return a copy. Blur affects only detected regions; this is not anonymization."""
    result = frame.copy()
    height, width = result.shape[:2]
    for box in boxes:
        x1, y1 = max(0, box.x), max(0, box.y)
        x2, y2 = min(width, box.x+box.width), min(height, box.y+box.height)
        if x2 <= x1 or y2 <= y1:
            continue
        if blur:
            region = result[y1:y2, x1:x2]
            small = cv2.resize(region, (max(1, (x2-x1)//16), max(1, (y2-y1)//16)))
            result[y1:y2, x1:x2] = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        else:
            cv2.rectangle(result, (x1, y1), (x2-1, y2-1), (80, 220, 120), 2)
    cv2.putText(result, f'Faces detected: {len(boxes)}', (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 120), 2, cv2.LINE_AA)
    return result

def process_image(source: Path, output: Path, detector: FaceDetector, *, blur: bool = False) -> dict:
    source, output = Path(source), Path(output)
    report_path = output.with_suffix('.json')
    if source.resolve() in (output.resolve(), report_path.resolve()):
        raise ValueError('Refusing to overwrite the input image.')
    if output.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
        raise ValueError('Output must end in .png, .jpg, or .jpeg.')
    if output.exists() or output.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError('Output already exists. Choose a new output name.')
    if not source.is_file():
        raise ValueError('Input image does not exist or is not a regular file.')
    if source.stat().st_size > 25 * 1024 * 1024:
        raise ValueError('Input exceeds the 25 MiB encoded-file limit.')
    # Native decoders are not a sandbox; only process images you trust.
    frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('OpenCV could not decode the input image.')
    boxes = detector.detect(frame)
    rendered = annotate(frame, boxes, blur=blur)
    ok, encoded = cv2.imencode(output.suffix.lower(), rendered)
    if not ok:
        raise RuntimeError('OpenCV could not encode the output image.')
    report = {'source_name': source.name, 'width': frame.shape[1], 'height': frame.shape[0],
              'detector': 'OpenCV Haar frontal-face cascade', 'face_count': len(boxes),
              'boxes': [asdict(box) for box in boxes], 'pixelated': blur,
              'warning': 'Detection is not recognition. Missed faces and false positives are possible.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation never overwrites an existing file, even on a repeat run.
    created = []
    try:
        with output.open('xb') as handle:
            created.append(output)
            handle.write(encoded.tobytes())
        with report_path.open('x', encoding='utf-8') as handle:
            created.append(report_path)
            json.dump(report, handle, indent=2)
            handle.write('\n')
    except OSError:
        # Clean up only the new files created by this invocation.
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return report
