# face-detection-lab 0.1.0

Face Detection Lab is a small local OpenCV application for frontal-face detection in images and live webcam frames.

It performs detection, not face recognition, identity matching, access control, attendance tracking, or surveillance.

## Highlights

- Detects frontal faces with OpenCV's bundled Haar cascade.
- Runs entirely locally without an account, API key, server, or model download.
- Produces annotated image output and a sibling JSON coordinate report.
- Supports optional pixelation of detected regions.
- Preserves source images and refuses to overwrite existing outputs.
- Includes live webcam preview without recording frames.

## Existing public validation

Public `main` commit `c36f38f3dbc4e745e97010348dbfb4459e4371ee` passed GitHub Actions run `36765353930`.

That run completed successfully across six hosted jobs:

- Ubuntu / Python 3.11
- Ubuntu / Python 3.13
- Ubuntu / Python 3.14
- macOS / Python 3.11
- macOS / Python 3.13
- macOS / Python 3.14

## Release-candidate validation

The release-candidate suite contains 16 tests after adding a package-version identity check.

The release-candidate workflow adds Python 3.12 on both Ubuntu and macOS, expanding the matrix from six to eight jobs.

Release publication requires the final `main` commit to pass the complete eight-job matrix.

A Mac local setup check succeeded with Python 3.14.6, OpenCV 4.13.0 and NumPy 2.3.5. `setup.sh` created the virtual environment, installed the pinned dependencies, and the 16-test release-candidate suite passed.

A hardware smoke test on camera index 0 successfully opened the live preview and returned cleanly to the shell. Terminal evidence does not independently establish whether a face box was visually observed, so this is treated as camera-access/preview evidence rather than a live-detection accuracy result.

## Scope

The bundled Haar detector can miss faces or produce false positives. This release makes no measured-accuracy claim.

Pixelation is not reliable anonymization.

Image decoding uses native OpenCV code; the application is intended for trusted local files rather than hostile public uploads.
