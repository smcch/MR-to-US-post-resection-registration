# Finite-element refinement of preoperative MR to post-resection intraoperative ultrasound registration

Code and landmarks of the paper

> S. Cepeda, O. Esteban-Sinovas, A. Šteňo, B. Hrebeňárová, I. Arrese, R. Sarabia.
> *Finite-element refinement of preoperative magnetic resonance to post-resection intraoperative ultrasound
> registration in brain tumor surgery.* Submitted to Medical Image Analysis, 2026.

The method is a deformable refinement stage applied after any initial alignment of the preoperative MR to the
intraoperative ultrasound (iUS) sweep acquired after tumor resection. A patient-specific linear-elastic
finite-element (FE) model built from the preoperative MR is used twice: its 30 lowest-energy modes form a
deformation space fitted to the iUS with the MIND-SSC similarity, and its stiffness turns block correspondences
into a smooth local displacement bounded to 2 mm. One configuration serves every case, and no landmark is used.

## Contents

| Path | What it is |
|---|---|
| `fe_refine.py` | **The proposed method**, run from file paths (preoperative MR, its SynthSeg label map, iUS sweep, initial affine). |
| `affine_init.py` | The initialization used in the paper: the NiftyReg affine baseline of the Learn2Reg 2024 ReMIND2Reg challenge. |
| `score_landmarks.py` | Target registration error (TRE) at landmark pairs for the initialization and for its refinement. |
| `fe_refinement/` | The modules of the method exactly as run for the paper (frozen configuration in `v2_method.py`). |
| `landmarks/ReMIND/` | The two-observer landmark set on 15 patients of the ReMIND collection (119 pairs), with its README. |
| `reproduce/` | The complete pipeline of the paper (every arm, comparator, statistic, table and figure) as it was run, and the per-case result tables. |
| `environment.yml` | The software environment of the paper's runs. |

## Installation

```bash
conda env create -f environment.yml
conda activate fe-refinement
```

The method needs a CUDA GPU with about 3 GB of memory. Two external tools are used around it:

- **NiftyReg 1.5.69** (`reg_aladin`) for `affine_init.py`; the conda-forge package is listed in `environment.yml`.
  Set `NIFTYREG_BIN` to its `bin` directory if it is not on the PATH.
- **SynthSeg** for the label map of the preoperative MR, with its robust mode:
  `mri_synthseg --i mr.nii.gz --o seg.nii.gz --robust` (FreeSurfer 7.4 or later, or the SynthSeg repository).
  The label map must be on the grid of the MR it was computed from.

## Running the method on one case

Inputs are NIfTI volumes in the navigation frame (the frame in which the tracked navigation system displays
the iUS over the preoperative MR). Voxels of the iUS outside the field of view of the sweep must be zero.

```bash
# 1. initialization: the challenge affine (iUS world -> MR world); --crop-fov when the sweep sits on a larger zero-padded grid
python affine_init.py --us us.nii.gz --mr mr.nii.gz --out work/affine --crop-fov

# 2. FE refinement from that initialization
python fe_refine.py --us us.nii.gz --mr mr.nii.gz --seg seg.nii.gz --affine work/affine/aff.txt --out work/fer

# 3. (optional) TRE at landmark pairs, before and after the refinement
python score_landmarks.py --mr-landmarks case-MR.mrk.json --us-landmarks case-US.mrk.json \
                          --affine work/affine/aff.txt --disp work/fer/disp_mm.nii.gz
```

`fe_refine.py` writes `disp_mm.nii.gz`, the refinement `u` on the 1 mm working grid of the iUS as a 3-vector field
in world millimeters, and `meta.json` with the regularity of the field (folding, SDLogJ) and the timings. The
complete transform maps an iUS point `x` (world mm) to `A (x + u(x))` in the MR, where `A` is the 4x4 matrix of
`--affine`. To display the preoperative MR in the frame of the iUS, sample the MR at `A (x + u(x))` for every
point `x` of the iUS grid. Any other initialization can be given as `--affine` (an identity matrix if the images
are already aligned, or a rigid pose from another method, as the paper does with the pose of MCPO).

On an NVIDIA RTX 3090 the FE model takes about 30 s (CPU) and the refinement about 40 s. Because the modal fit
runs on the GPU, repeated runs differ slightly (the paper reports a second run: case-mean TRE within 0.02 mm).

## Reproducing the ReMIND results of the paper

1. Download the ReMIND collection from The Cancer Imaging Archive (Juvekar et al., *Sci. Data* 2024). For each of
   the 15 patients, `landmarks/ReMIND/images.csv` names the preoperative T2-weighted series used as the moving
   image and the iUS series (`USpreimri`, the three-dimensional sweep acquired after resection and before the
   intraoperative MR). Convert them to NIfTI keeping the scanner coordinates of the DICOM headers (for example with
   `dcm2niix`); in ReMIND the iUS and the preoperative MR share the navigation frame.
2. Compute the SynthSeg label map of the T2, the challenge affine (`affine_init.py --crop-fov`) and the refinement
   (`fe_refine.py`), then score with `score_landmarks.py` and the files of `landmarks/ReMIND/landmarks/`.
3. The expected values are those of Supplementary Table S1 of the paper (also in
   `reproduce/tables/PER_CASE.csv`). As an example, ReMIND-024 gives a case-mean TRE of 3.88 mm after the affine and
   2.38 mm after the refinement; this repository's entry points were checked to reproduce the paper's run of that
   case to 0.02 mm in the displacement field.

RESECT (Xiao et al. 2017) and BITE (Mercier et al. 2012) are public and come with their own landmarks; the scripts
in `reproduce/` show how they were used. The images and landmarks of the external site of the paper are not
redistributable and are not in this repository.

## Landmarks

`landmarks/ReMIND/` contains 119 homologous landmark pairs on 15 ReMIND patients, each pair annotated on the
preoperative T2 and on the post-resection iUS, placed by two neurosurgeons and consolidated by the rules
described in `landmarks/ReMIND/README.md` and in the paper (Section 2 and Supplementary Section S6). The files are
3D Slicer markups (LPS coordinates) plus one CSV table. They are released under CC BY-NC 4.0.

## The full pipeline of the paper

`reproduce/scripts/` holds the scripts that produced every number of the paper: the refinement on all cohorts
(`fer_run.py`), the comparators (`fer_f3d.py`, `fer_extra_deformables.py` for ConvexAdam, deeds and SyN,
`fer_mcpo.py` and `fer_mcpo_pose.py` for MCPO and its pose), the scoring and statistics (`fer_score.py`,
`fer_steps.py`, `fer_sensitivity.py`, `fer_verify_affines.py`), the two-observer landmark consensus
(`landmarks_final_build.py`) and the observer-variability analyses. `reproduce/paper_tools/` holds the generators
of the paper's numbers, tables and figures, and `reproduce/tables/` the per-case and per-landmark result tables.
These scripts are provided as they were run, with the paths of the authors' machine; see `reproduce/README.md`.
MCPO is not included: the paper runs the authors' official script (github.com/wjiazheng/MCPO) at its defaults.

## Citation

If you use the code or the landmarks, please cite the paper above (a preprint reference will be added here) and,
for the landmarks, also the ReMIND collection: Juvekar P. et al., *ReMIND: the brain resection multimodal imaging
database*, Sci. Data 11, 494 (2024).

## License

The code, the documentation and the landmarks are released for non-commercial use under the Creative Commons
Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0; see `LICENSE`). Commercial use requires the
permission of the authors. This is research software and research data, not a medical device.
