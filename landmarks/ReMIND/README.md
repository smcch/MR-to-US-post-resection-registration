# Two-observer landmarks: preoperative MR and post-resection intraoperative ultrasound (ReMIND)

Homologous anatomical landmark pairs for 15 patients of the ReMIND collection (Brain Resection Multimodal
Imaging Database), for evaluating the registration of the preoperative MR to the intraoperative ultrasound
(iUS) sweep acquired after resection. The set contains 119 landmark pairs (6 to 10 per patient).

## Images
For each patient, `images.csv` lists the ReMIND series the coordinates refer to: the preoperative T2-weighted
series (moving image) and the post-resection iUS sweep acquired before intraoperative MR (`USpreimri`).

## Coordinates and files
- `landmarks/<case>-MR.mrk.json` and `landmarks/<case>-US.mrk.json`: 3D Slicer markups files. Points with the
  same label number (`MR-k`, `US-k`) form a pair. Numbers are not consecutive because landmarks removed during
  consensus keep their original identifiers.
- `landmarks_ReMIND_LPS.csv`: the same pairs in one table.
- All coordinates are world (patient) coordinates in millimeters in the LPS convention, as stored in the image
  headers and used by 3D Slicer and DICOM.

## Annotation
Each pair was defined on point-like features identifiable in both modalities: sulcal fundi and corners, tips
of the ventricular horns, vessel-related echogenic points, and falx and tentorium junctions. The MR point was
placed once on the preoperative T2 by observer 1. Its iUS counterpart was placed independently by two observers in
3D Slicer, observer 1 in two sessions and observer 2 in one session, with the images in the unregistered navigation
alignment. Both observers are neurosurgeons: observer 1 has 15 years of experience in intraoperative ultrasound and
image analysis, observer 2 has 3 years of experience.

## Consensus
1. Ten pairs were annotated per patient. When two MR points were less than 3 mm apart, they were treated as the
   same feature and only the lower-numbered pair was kept.
2. An iUS point lying more than 5.79 mm from both other iUS points of the same pair was removed. The threshold
   is the 99.9th percentile of a Maxwell distribution fitted to observer 1's repeat placements and was fixed
   before observer 2's placements were analyzed.
3. A pair was kept when its remaining iUS points came from both observers and lay within 5.79 mm of each other.
4. The released iUS point is the mean of the two observers' positions, with observer 1's position taken as the
   mean of the sessions kept. The released MR point is the original MR point.

## Observer variability (iUS points)
Mean over patients of the per-patient mean distance, ± standard deviation over patients, in mm. "Released pairs" uses
only the placements averaged into the released points.

| Comparison | Before consensus | Released pairs |
|---|---|---|
| Intra-observer: observer 1, session 1 vs session 2 | 2.83 ± 0.72 | 2.19 ± 0.58 |
| Inter-observer: observer 2 vs the mean of observer 1's sessions | 2.88 ± 0.76 | 2.25 ± 0.31 |
| Observer 2 vs observer 1, session 1 | 3.19 ± 0.71 | 2.57 ± 0.34 |
| Observer 2 vs observer 1, session 2 | 3.25 ± 1.03 | 2.39 ± 0.50 |

## License and citation
The landmarks are released under the Creative Commons Attribution-NonCommercial 4.0 International license
(CC BY-NC 4.0) and deposited in Zenodo (https://doi.org/10.5281/zenodo.24008945). Please cite the paper named in
the repository README, the Zenodo record (Cepeda S., Esteban Sinovas O., Expert-annotated landmarks for registration
of preoperative MRI to post-resection intraoperative ultrasound in the ReMIND dataset [dataset], Zenodo, v1, 2026)
and the ReMIND collection (Juvekar P. et al., ReMIND: the brain resection multimodal imaging database, Sci. Data 11,
494, 2024).
