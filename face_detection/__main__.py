"""Run with python -m face_detection image ... or webcam."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import cv2
from .core import FaceDetector, annotate, process_image

def webcam(detector: FaceDetector, camera: int, blur: bool) -> None:
    if camera < 0:
        raise ValueError('Camera index must be nonnegative.')
    capture = cv2.VideoCapture(camera)
    try:
        if not capture.isOpened():
            raise RuntimeError('Cannot open camera. Allow camera access for your terminal application; try --camera 1.')
        previous = time.perf_counter()
        print('Camera is active. Press Q or Esc in the preview window to exit. Nothing is recorded.')
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError('Camera stopped returning frames.')
            boxes = detector.detect(frame)
            rendered = annotate(frame, boxes, blur=blur)
            now = time.perf_counter()
            fps = 1.0 / max(now-previous, 1e-6)
            previous = now
            cv2.putText(rendered, f'{fps:.1f} FPS | Q to exit', (12, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (80, 220, 120), 1, cv2.LINE_AA)
            cv2.imshow('Face Detection Lab', rendered)
            if cv2.waitKey(1) & 0xff in (ord('q'), ord('Q'), 27):
                break
            if cv2.getWindowProperty('Face Detection Lab', cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Local face detection, without recognition or cloud uploads.')
    sub = parser.add_subparsers(dest='command', required=True)
    image = sub.add_parser('image', help='Annotate a trusted local image and write a JSON report.')
    image.add_argument('input', type=Path)
    image.add_argument('--output', type=Path, required=True)
    camera = sub.add_parser('webcam', help='Live preview; frames are never saved.')
    camera.add_argument('--camera', type=int, default=0)
    for command in (image, camera):
        command.add_argument('--blur', action='store_true', help='Pixelate detected regions; not reliable anonymization.')
        command.add_argument('--scale-factor', type=float, default=1.1)
        command.add_argument('--min-neighbors', type=int, default=5)
    args = parser.parse_args(argv)
    try:
        detector = FaceDetector(scale_factor=args.scale_factor, min_neighbors=args.min_neighbors)
        if args.command == 'image':
            report = process_image(args.input, args.output, detector, blur=args.blur)
            print(f"Detected {report['face_count']} face(s).")
            print(f'Image: {args.output}\nReport: {args.output.with_suffix(".json")}')
        else:
            webcam(detector, args.camera, args.blur)
    except KeyboardInterrupt:
        print('\nStopped. No webcam frames were saved.')
        return 130
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
