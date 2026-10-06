# Design decisions

A short log of the decisions made in this project, each with its reason and the
numbers it is based on. Numbers come from `reports/dataset_report.md`, created
with `python -m detection.dataset_check` on the original Roboflow split.

## Dataset

**Choice:** [3d print error box](https://universe.roboflow.com/3d-test/3d-print-error-box),
version 26, licence CC BY 4.0, exported in YOLOv8 format. It replaces a first
dataset whose labels turned out to be unusable (see "Rejected first dataset").

**Why version 26:** it is the newest version without Roboflow augmentation; the
only preprocessing is auto-orient. Augmented versions contain two or three
altered copies of each training image, which would inflate the image count and
the duplicate search. Augmentation is done by Ultralytics during training instead.

| | train | valid | test | total |
|---|---:|---:|---:|---:|
| Images | 2086 | 284 | 254 | 2624 |
| Background images (no defect) | 671 | 75 | 70 | 816 |
| Boxes | 2337 | 281 | 207 | 2825 |

**Data quality:** no images without labels, no labels without images and no
unreadable images. 11 label lines in 6 training files are polygons instead of
boxes. Ultralytics turns polygon files into boxes on its own; in one file
polygons and boxes are mixed, so 2 boxes of that image are converted wrongly.
This affects 1 of 2624 images and is accepted.

**Background images are kept.** 31 % of the images show prints without a labelled
defect. They teach the model what a good print looks like, which reduces false
alarms.

**Image sizes vary** (697 different sizes in train, most often 1080x1440). YOLO
scales every image to 640 pixels on its longest side, so small boxes in large
images become smaller still.

**Label check:** box sizes per class, as a share of the image area. The values
fit the defects: spaghetti covers large areas, warping is a lifted corner or edge.
Sample images with their boxes were also checked by eye.

| Class | Boxes | Median area | Below 1 % | 10 % or more |
|---|---:|---:|---:|---:|
| `layer shift` | 501 | 2.6 % | 24 % | 16 % |
| `spaghetti` | 735 | 16.8 % | 5 % | 61 % |
| `stringing` | 893 | 3.6 % | 18 % | 25 % |
| `warping` | 696 | 1.1 % | 48 % | 4 % |

## Classes

| Class ID | Name | Meaning | Boxes (total / test) |
|---:|---|---|---:|
| 0 | `layer shift` | Layers offset sideways, the part looks cut and shifted | 501 / 37 |
| 1 | `spaghetti` | Tangled filament, the print has come loose | 735 / 54 |
| 2 | `stringing` | Thin threads between parts of the print | 893 / 47 |
| 3 | `warping` | Corners bent upwards, part lifting off the bed | 696 / 69 |

All four classes are kept unchanged. The smallest class has 501 boxes, far above
the threshold of 30 from the specification, so no class has to be merged or
removed. Expected difficulty: `warping`, because half of its boxes cover less
than 1 % of the image.

## Rejected first dataset

The project started with [3D Print Defect](https://universe.roboflow.com/shahrilspace/3d-print-defect)
(version 1, 470 images, classes `defect`, `spaghetti`, `under extrusion layer`,
`warping`). Its data passed all formal checks, but the first training
(`configs/baseline.yaml`, yolov8n, 50 epochs, seed 42, Tesla T4) reached only:

| | Best epoch (32) | Last epoch (50) |
|---|---:|---:|
| mAP50 | 0.056 | 0.006 |
| Recall | 0.03 | 0.03 |

The cause was the labels, not the training. Spaghetti clumps that fill half of
the image were marked with a tiny box at an arbitrary spot inside them:

| Class | Median box area | Boxes below 1 % of the image |
|---|---:|---:|
| `spaghetti` | 0.8 % | 64 % |
| `defect` | 1.6 % | 25 % |

With such labels the model cannot learn where a box belongs, and no training
setting fixes that. In addition, `defect` was also used for small surface flaws
such as zits, not only for total failures. Since both the original and the clean
split would end near zero, the split comparison, the core result of the project,
could not be shown with this dataset. The lesson: formal label checks (format,
value range) are not enough; box sizes per class and a look at sample images
belong to every dataset check.

The duplicate search on this dataset found that 8 of 46 test images (17 %) had a
near twin in train; this is where the method below was developed.

## Near-duplicate images

**Method:** every image gets a perceptual hash (`phash`, 64 bit). Two images count
as twins when their hashes differ in at most **14 bits**, and twins of twins form
one group (`python -m detection.grouping`, report in `reports/grouping_report.md`).

**Why 14** (measured on the first dataset): the distance from each image to its
closest other image split into two clear clusters: 80 images at 0 to 4 bits and
almost all others at 16 bits or more. The few pairs in between were checked by eye:

| Distance | Pairs checked | Same scene |
|---:|---:|---:|
| 4 to 12 | 6 | 6 |
| 14 | 4 | 3 |
| 16 | 8 (sample of 24) | 0 |

A missed twin causes leakage, while a wrong match only makes one group larger,
so the limit is set at the upper end, 14.

**Open:** the limit has to be checked again on the new dataset, and the results
of the duplicate search are measured anew.

**Limitation:** phash only finds images that look alike as a whole. Two frames of
the same printer filmed at different moments can be further apart than 14 bits
and are then not grouped.

## Clean split

**Ratio:** 70 / 15 / 15 (train / valid / test), as in the specification.

**Method:** every group of near-identical images goes as a whole into one split
(`python -m detection.resplit`, seed 42). The split is stratified by class: each
group gets the rarest class among its boxes as its main class, and the groups of
each main class are divided in the target ratio on their own. Without this, a
rare class could end up with almost no test examples by chance.

**Open:** the clean split is created anew for the new dataset.
