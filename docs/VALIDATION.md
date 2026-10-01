# Validation record

## Public implementation baseline

Repository: `dexter02-crypt/face-detection-lab`

Baseline `main` commit:

`c36f38f3dbc4e745e97010348dbfb4459e4371ee`

GitHub Actions run `36765353930` completed successfully on that exact commit.

The six successful jobs were:

- macOS / Python 3.11
- macOS / Python 3.13
- macOS / Python 3.14
- Ubuntu / Python 3.11
- Ubuntu / Python 3.13
- Ubuntu / Python 3.14

Each hosted job installed the declared OpenCV/NumPy dependencies and ran the test suite.

## Release-candidate coverage

The release-candidate suite contains **16 tests**.

Coverage includes:

- real bundled-cascade inference on the supplied NASA sample
- blank-image negative behavior
- invalid frame and detector-configuration rejection
- annotation input preservation
- detected-region pixelation
- coordinate mapping after detection-image resizing
- real image and JSON output creation
- refusal to overwrite existing outputs
- refusal to overwrite the source image
- existing JSON-report preservation
- invalid image rejection
- real image-processing CLI execution
- clean CLI error handling
- webcam failure cleanup and camera release
- package version identity for 0.1.0

The release-candidate workflow adds Python 3.12 on Ubuntu and macOS, expanding the hosted matrix to eight jobs.

Fresh remote CI is required before release.

## Demonstration

The bundled Haar classifier detects a face in the included 512 × 512 NASA sample. The repository includes `docs/demo.png` and `docs/demo.json` as generated outputs.

This demonstrates one known example only. It is not an accuracy benchmark.

## Hardware boundary

Automated CI does not establish successful access to a physical webcam.

The webcam cleanup test uses a mocked failed camera and verifies resource release.

## Mac local evidence

On the maintainer Mac, `setup.sh` succeeded using Python 3.14.6 with OpenCV 4.13.0 and NumPy 2.3.5. The 16-test release-candidate suite passed in that virtual environment.

A hardware smoke test using camera index 0 successfully opened the live preview and returned cleanly to the shell.

The terminal record does not independently show whether a detection box appeared on the live face, so the webcam result establishes camera access and preview operation only. It is not a live-detection accuracy benchmark and is not generalized to other hardware.

## Scope boundaries

This is face detection, not recognition.

The application makes no measured detection-accuracy, identity, surveillance, biometric, or anonymization guarantee.

Pixelation affects only detected boxes and is not reliable privacy protection.

Encoded image input is bounded to 25 MiB, but image decoding is native-code processing and is not a hostile-file sandbox.
