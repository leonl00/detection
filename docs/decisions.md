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

**Check on the current dataset:** here the distances are less clearly separated.
Many images come from the same fixed camera at the same printer, often with
different prints on the bed. Pairs were checked by eye again (6 per distance):

| Distance | Pairs at this distance | What the pairs show |
|---:|---:|---|
| 10 | 68 | almost all the same camera setup, partly the same print |
| 12 | 108 | almost all the same camera setup |
| 14 | 249 | about half the same camera setup |
| 16 | 879 | mostly unrelated images |

Because twins of twins form one group, a higher limit can chain unrelated images
into large groups. This was measured as well:

| Limit | Groups | Largest group | Images in groups of 2 or more |
|---:|---:|---:|---:|
| 12 | 2356 | 59 | 424 |
| 14 | 2236 | 75 | 611 |
| 16 | 1637 | 143 | 1356 |

The limit of **14 is kept**. The largest group (75 images) was checked: 15 of 16
sample images show the same printer and camera, so the group is real and small
enough (3 % of the data) to be put into one split as a whole. At 16, unrelated
images would be chained together.

**Result on the original Roboflow split:**

| Question | Images |
|---|---:|
| Test images with a twin in train | 64 of 254 (25 %) |
| Valid images with a twin in train | 67 of 284 (24 %) |
| Test images with a twin in valid | 19 of 254 (7 %) |

2624 images form 2236 groups; 223 groups contain more than one image, and 104
of them are spread over more than one split. A quarter of the original test set
shows scenes that are also in train, so its metrics are expected to be too
optimistic.

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

**Result** (`data/dataset_clean/`, checked with `dataset_check` and `grouping`):

| | train | valid | test |
|---|---:|---:|---:|
| Images | 1837 | 394 | 393 |
| Background images | 602 | 107 | 107 |
| `layer shift` boxes | 346 | 75 | 80 |
| `spaghetti` boxes | 528 | 104 | 103 |
| `stringing` boxes | 642 | 129 | 122 |
| `warping` boxes | 461 | 123 | 112 |
| Test images with a twin in train | | | 0 of 393 |

The clean test set is also larger than the original one (393 instead of 254
images), because Roboflow used about 80 / 11 / 10.

## Evaluation

**Method:** `python -m detection.evaluate` runs the Ultralytics validation on the
test split (mAP50, mAP50-95, precision and recall per class, confusion matrix,
precision-recall curve) and adds an error analysis per image. A prediction is a
hit when it has the same class as a true box and overlaps it by IoU >= 0.5.
Leftover true boxes are missed defects, leftover predictions are false alarms.
The 20 images with the most errors are drawn; on a tie, more missed defects rank
first, because a missed defect costs a failed print while a false alarm only
costs a look at the printer. The error analysis uses the Ultralytics default
confidence of 0.25; the threshold for the application is chosen in stage 8.

**First result:** baseline model (`runs/baseline`, trained on the original
split), evaluated on the original test split (254 images, 207 boxes):

| Class | Boxes | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| **all** | 207 | 0.623 | 0.577 | 0.601 | 0.335 |
| `layer shift` | 37 | 0.440 | 0.243 | 0.245 | 0.103 |
| `spaghetti` | 54 | 0.759 | 0.814 | 0.848 | 0.503 |
| `stringing` | 47 | 0.637 | 0.787 | 0.791 | 0.522 |
| `warping` | 69 | 0.657 | 0.464 | 0.520 | 0.212 |

A quarter of these test images have a near twin in train, so the numbers are
expected to be too optimistic; the comparison with the clean split follows in
stage 7. `layer shift` is clearly the weakest class: only 24 % of its boxes are
found.

**Observations from the 20 worst images:**

- Several "worst" images are disagreements about box boundaries, not missed
  defects: the label has one large box around a spaghetti clump or a group of
  strings, the model finds the same defect as several smaller boxes. This counts
  as one missed defect plus several false alarms, although the defect is found.
- Warping boxes along a lifted edge often overlap the label only partly.
- Real false alarms also occur, for example boxes on a keyboard next to the print.

The box-level metrics therefore understate how often the model notices that a
print has a problem. For the application, which only has to raise an alarm,
an image-level view (is there a defect in the image or not?) is checked in stage 8.
