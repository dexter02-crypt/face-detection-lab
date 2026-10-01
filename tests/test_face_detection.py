import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import face_detection
import cv2
import numpy as np
from face_detection.core import FaceBox, FaceDetector, annotate, process_image

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'examples' / 'astronaut.png'
class FaceDetectionTests(unittest.TestCase):
    def test_package_version_matches_release(self):
        self.assertEqual(face_detection.__version__, "0.1.0")

    @classmethod
    def setUpClass(cls):
        cls.detector = FaceDetector()

    def test_real_cascade_detects_sample_face(self):
        boxes = self.detector.detect(cv2.imread(str(SAMPLE)))
        # Actual cascade inference. No classifier mock or fabricated detections.
        self.assertTrue(any(150 <= b.x <= 210 and 40 <= b.y <= 100 and 70 <= b.width <= 130 for b in boxes), boxes)

    def test_blank_image_has_no_faces(self):
        self.assertEqual(self.detector.detect(np.zeros((256,256,3), dtype=np.uint8)), [])

    def test_invalid_frames(self):
        for frame in (None, np.zeros((1,1,3)), np.zeros((0,0,3), dtype=np.uint8), np.zeros((10,10), dtype=np.uint8)):
            with self.subTest(frame_type=type(frame)), self.assertRaises(ValueError):
                self.detector.detect(frame)

    def test_invalid_configuration(self):
        for kwargs in ({'scale_factor':1.0}, {'scale_factor':float('nan')}, {'min_neighbors':0}, {'min_size':0}, {'max_dimension':10}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FaceDetector(**kwargs)

    def test_annotation_leaves_input_unchanged(self):
        image = cv2.imread(str(SAMPLE)); original = image.copy()
        result = annotate(image, [FaceBox(170,60,100,100)])
        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.array_equal(result, original))

    def test_pixelation_changes_detected_region(self):
        image = cv2.imread(str(SAMPLE))
        result = annotate(image, [FaceBox(170,60,100,100)], blur=True)
        self.assertFalse(np.array_equal(result[60:160,170:270], image[60:160,170:270]))
        np.testing.assert_array_equal(result[200:300,200:300], image[200:300,200:300])

    def test_large_image_coordinates_map_back(self):
        image = cv2.resize(cv2.imread(str(SAMPLE)), (1536,1536))
        boxes = FaceDetector(max_dimension=512).detect(image)
        self.assertTrue(any(450 <= b.x <= 630 and 120 <= b.y <= 300 for b in boxes))
        self.assertTrue(all(b.x+b.width <= 1536 and b.y+b.height <= 1536 for b in boxes))

    def test_image_and_json_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'out.png'
            report = process_image(SAMPLE, output, self.detector)
            self.assertGreaterEqual(report['face_count'], 1)
            self.assertIsNotNone(cv2.imread(str(output)))
            self.assertEqual(json.loads(output.with_suffix('.json').read_text()), report)
            self.assertEqual(report['source_name'], 'astronaut.png')

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'out.png'; output.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                process_image(SAMPLE, output, self.detector)
            self.assertEqual(output.read_bytes(), b'keep')

    def test_input_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):
            process_image(SAMPLE, SAMPLE, self.detector)

    def test_existing_report_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'out.png'; output.with_suffix('.json').write_text('keep')
            with self.assertRaises(FileExistsError):
                process_image(SAMPLE, output, self.detector)
            self.assertFalse(output.exists())

    def test_invalid_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad=Path(tmp)/'bad.png'; bad.write_text('not an image')
            with self.assertRaises(ValueError):
                process_image(bad, Path(tmp)/'out.png', self.detector)

    def test_cli_runs_real_image_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-m','face_detection','image',str(SAMPLE),'--output',str(Path(tmp)/'out.png')],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('Detected ',result.stdout)
            self.assertTrue((Path(tmp)/'out.json').is_file())

    def test_cli_reports_bad_input_without_traceback(self):
        result=subprocess.run([sys.executable,'-m','face_detection','image','missing.png','--output','outputs/missing.png'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('Error:',result.stderr)
        self.assertNotIn('Traceback',result.stderr)

    def test_camera_failure_releases_device(self):
        from face_detection.__main__ import webcam
        with patch('face_detection.__main__.cv2.VideoCapture') as capture, patch('face_detection.__main__.cv2.destroyAllWindows'):
            capture.return_value.isOpened.return_value=False
            with self.assertRaises(RuntimeError):
                webcam(self.detector,0,False)
            capture.return_value.release.assert_called_once()
