# Validation record

## Executed here

- Environment: Linux, CPython 3.13.5.
- Command: `python -B -m unittest discover -s tests -v`.
- Result: **15 tests passed**, zero failed, zero skipped.
- Full captured test output: [test-output.txt](test-output.txt).

The environment metadata records `opencv-python==4.13.0.92`; imported `cv2` reports 4.13.0 and NumPy reports 2.3.5. The base image also contains headless-OpenCV package metadata, so this is not evidence of an isolated desktop/GUI package installation.

A clean dependency-download installation was not verified: the build environment could not reach the package index. Your Mac setup is the remaining installation check.

## Actual demonstration

The real bundled Haar classifier detected one face in the supplied 512 × 512 NASA sample. `docs/demo.png` and `docs/demo.json` are actual application outputs, not a mockup. A blank image also passes the negative test. The webcam failure-cleanup test uses a fake camera; no successful live-camera run was performed.

## Boundaries

macOS installation, your local GUI/file-opening behavior, successful live webcam
capture, real GitHub publication and GitHub Actions execution have **not** been
verified by this record. The workflow requests multiple Python/OS combinations;
that configuration is not evidence that those jobs ran. No measured detection
accuracy, production-readiness or universal input-correctness claim is made.
These are author-run tests, not independent certification.

## Your local verification

Run setup and the sample on your own machine. Once the repository is published,
record your actual OS/Python versions, the command, its real result, and one
small change you understand. Do not rewrite unexecuted checks as passing checks.
