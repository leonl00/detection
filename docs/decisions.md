# Design decisions

A short log of the decisions made in this project, each with its reason and the
numbers it is based on. Numbers come from `reports/dataset_report.md`, created
with `python -m detection.dataset_check` on the original Roboflow split.

## Dataset

**Choice:** [3D Print Defect](https://universe.roboflow.com/shahrilspace/3d-print-defect),
version 1, licence CC BY 4.0, exported in YOLOv8 format.

| | train | valid | test | total |
|---|---:|---:|---:|---:|
| Images | 331 | 93 | 46 | 470 |
| Boxes | 352 | 97 | 46 | 495 |

**Data quality:** no images without labels, no labels without images, no invalid
label lines and no unreadable images. All images are 640x640 pixels.

**Observation:** about one box per image, and 41 % of all boxes (205 of 495) cover
less than 1 % of the image area. Small objects are the hardest case for YOLO, so
the image size stays at 640 and is not reduced to save training time.

**Observation:** the file names (for example `beautiful_failures`, `failed-prints-2`,
`666x500`) suggest photos collected from the web rather than video frames. The
main leakage risk is therefore the same photo appearing more than once, which the
duplicate search in stage 4 checks.

## Classes

| Class ID | Name | Meaning | Boxes (total / test) |
|---:|---|---|---:|
| 0 | `defect` | Total failure: the whole part is unusable | 161 / 17 |
| 1 | `spaghetti` | Tangled filament above the part | 217 / 19 |
| 2 | `under extrusion layer` | Gaps and thin layers from too little material | 81 / 6 |
| 3 | `warping` | Corners bent upwards, part lifting off the bed | 36 / 4 |

**`defect` is kept as its own class** with the meaning "total failure of the
whole part", while the other three classes describe local defect patterns. The
name in the data stays unchanged so that the original and the clean split use
exactly the same classes. Expected risk: confusion between `defect` and
`spaghetti`, because a spaghetti print is often a total failure as well. The
confusion matrix in stage 6 checks this.

**`warping` is kept** although it is the smallest class. With 36 boxes it is
above the threshold of 30 from the specification, and warping is a typical FDM
defect. Limitation: the original test split contains only 4 warping boxes, so
one hit more or less changes its recall by 25 percentage points. Results for
this class are reported with this uncertainty, and comparisons use three seeds.

## Near-duplicate images

**Method:** every image gets a perceptual hash (`phash`, 64 bit). Two images count
as twins when their hashes differ in at most **14 bits**, and twins of twins form
one group (`python -m detection.grouping`, report in `reports/grouping_report.md`).

**Why 14:** the distance from each image to its closest other image splits into
two clear clusters: 80 images at 0 to 4 bits and almost all others at 16 bits or
more. The few pairs in between were checked by eye:

| Distance | Pairs checked | Same scene |
|---:|---:|---:|
| 4 to 12 | 6 | 6 |
| 14 | 4 | 3 |
| 16 | 8 (sample of 24) | 0 |

A missed twin causes leakage, while a wrong match only makes one group larger,
so the limit is set at the upper end, 14.

**Result on the original Roboflow split:**

| Question | Images |
|---|---:|
| Test images with a twin in train | 8 of 46 (17 %) |
| Valid images with a twin in train | 15 of 93 (16 %) |
| Test images with a twin in valid | 1 of 46 (2 %) |

470 images form 420 groups; 43 groups contain more than one image (largest: 3),
and 22 of them are spread over more than one split. The test metrics of the
original split are therefore partly measured on scenes the model has seen.

**Limitation:** phash only finds images that look alike as a whole. Two frames of
the same printer filmed at different moments can be further apart than 14 bits
and are then not grouped.

## Clean split

**Ratio:** 70 / 15 / 15 (train / valid / test), as in the specification. The test
set grows from 46 to about 70 images, which makes per-class results less noisy.

**Method:** every group of near-identical images goes as a whole into one split
(`python -m detection.resplit`, seed 42). The split is stratified by class: each
group gets the rarest class among its boxes as its main class, and the groups of
each main class are divided in the target ratio on their own. Without this, a
rare class such as `warping` could end up with almost no test examples by chance.

**Result** (`data/dataset_clean/`, checked with `dataset_check` and `grouping`):

| | train | valid | test |
|---|---:|---:|---:|
| Images | 329 | 71 | 70 |
| `defect` boxes | 112 | 25 | 24 |
| `spaghetti` boxes | 155 | 32 | 30 |
| `under extrusion layer` boxes | 55 | 14 | 12 |
| `warping` boxes | 24 | 6 | 6 |
| Test images with a twin in train | | | 0 of 70 |
