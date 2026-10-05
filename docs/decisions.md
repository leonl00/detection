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
