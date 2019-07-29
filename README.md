# cosmobot-deep-learning

Cosmobot deep learning models and helper code

## Iterating on models

### When to branch

This repo will contain one directory for each major type of model, with the best version of each checked into master.

If you are making tweaks to a model, do so in a branch. If the tweaks turn out to be an improvement on that model, land them. If you are creating an new type of model (rather than iterating on an existing model), create a new directory for it.

### Documenting changes

Create a directory in the Experiments directory on Google Drive with a README detailing your changes and results.

Reference your branch name or changeset in the README.

### Iterating with Jupyter notebooks

TODO: [https://app.asana.com/0/819671808102776/1130875537031890/f]()


## Terminology
Some standard terminology around our raw image data and how we process it

COPY-PASTA: These definitions have been copied from the cosmobot-process-experiment repo

* `RAW image file` - A JPEG+RAW image file as directly captured by a PiCam v2, saved as a .JPEG
* `RGB image` - A 3D numpy.ndarray: a 2D array of "pixels" (row-major), where each "pixel" is a 1D array of [red, green, blue] channels with a value between 0 and 1. This is our default format for interacting with images. An example 4-pixel (2x2) image would have this shape:

```
[
 [ [r1, g1, b1], [r2, g2, b2] ],
 [ [r3, g3, b3], [r4, g4, b4] ]
]
```

* `ROI` - An `RGB image` that has been cropped to a specific Region of Interest (ROI).
* `ROI definition` - A 4-tuple in the format provided by cv2.selectROI: (start_col, start_row, cols, rows), used to define a Region of Interest (ROI).
