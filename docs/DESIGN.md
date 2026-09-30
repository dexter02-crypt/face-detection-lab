# Design notes

The detector, rendering, and file-writing functions are separate so tests can
exercise inference independently from output and camera hardware.

Haar cascades were chosen because OpenCV ships the model, inference runs on a
CPU, and the repository does not need a model download or training pipeline.
This is a simplicity decision, not a claim that Haar is the best modern face
detector. A later, separate comparison could evaluate another detector on a
properly licensed test set.

The detection image is bounded by its longest dimension. Separate horizontal
and vertical ratios map rectangles back after integer resize rounding.
Annotations are drawn on a copy. Image and JSON files use exclusive creation,
and the source cannot be the destination. Camera acquisition is enclosed in a
`try/finally` so errors still release the device. There is no video recorder.

Three questions to explain in an interview: why use grayscale; what happens
when min-neighbors increases; why a passing single-image test is not an
accuracy benchmark. A small personal extension would be a side-by-side
parameter comparison on a consented image, with observations documented.
