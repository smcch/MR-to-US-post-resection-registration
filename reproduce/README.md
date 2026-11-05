# The pipeline of the paper, as it was run

These scripts produced every number, table and figure of the paper. They are provided for inspection and
adaptation, not as a turnkey tool: they read and write fixed paths of the authors' machine
(`/mnt/c/ReMIND_work/...`, with a few references to an archive drive `/mnt/g/...`), they expect the cohorts to be
staged in that layout, and some of them call external programs (NiftyReg, SynthSeg, ConvexAdam, deedsBCV, ANTs,
and the official MCPO script, which is not included). For running the proposed method on new data use
`fe_refine.py` at the root of the repository.

## Map

| Script | Role in the paper |
|---|---|
| `scripts/fer_common.py` | Cohort lists, input paths, landmark readers, shared helpers of the `fer_*` family (`LANDMARK_SET = "final"` selects the released two-observer landmarks). |
| `scripts/fer_run.py` | The FE refinement on every registration from the challenge affine (Section 3, Results 5.1). |
| `scripts/fer_mcpo_pose.py`, `scripts/mcpo_pose_deform.py` | The rigid pose fitted to MCPO's field and the same refinement from it (Results 5.3). |
| `scripts/fer_f3d.py` | NiftyReg free-form stage on the same affine (MIND-SSC and LNCC variants). |
| `scripts/fer_extra_deformables.py`, `scripts/_syn_worker.py`, `scripts/fer_extra_summary.py` | ConvexAdam, deeds, SyN and the cropped free-form stage on the same affine, with the field-of-view treatments of Supplementary Section S3 (Results 5.2). |
| `scripts/fer_mcpo.py` | MCPO, the authors' official script at its defaults, from the navigation alignment (Results 5.5). |
| `scripts/fer_score.py` | Per-landmark errors of every arm, per-case table (`tables/PER_CASE.csv`, `tables/LANDMARKS.csv`). |
| `scripts/fer_steps.py` | Second run of the refinement and the two steps run alone (Results 5.7, Supplementary S11). |
| `scripts/fer_sensitivity.py`, `scripts/fer_verify_affines.py`, `scripts/fer_fix_affines.py` | Sensitivity to the constants (S8); reproducibility of the challenge affine (S5). |
| `scripts/l2r_baseline_cohorts.py`, `scripts/l2r_f3d_cohorts.py`, `scripts/niftyreg_prep.py` | The challenge affine and free-form baselines on the public cohorts. |
| `scripts/rg_cohorts.py`, `scripts/resect_tags.py`, `scripts/bite_tags.py`, `scripts/single_config.py`, `scripts/method_arms.py` | Cohort definitions and landmark readers for RESECT and BITE. |
| `scripts/v2_method.py`, `scripts/v2_deform.py`, `scripts/reg_core.py`, `scripts/reg_modal.py`, `scripts/biomech_basis.py` | The method (same files as `fe_refinement/`). |
| `scripts/v2_remind_eval.py`, `scripts/affine_cascade.py`, `scripts/rs_prestage.py`, `scripts/rigid_*.py` | Older evaluation and pose code on which `fer_run.py` depends for a few functions. |
| `scripts/brat_*.py`, `scripts/brat_*.sh` | Staging and processing of the Bratislava cases (data not redistributable). |
| `scripts/fer_prepare_remind.py`, `scripts/stage_remind_new.sh` | Staging of ReMIND cases (affine, working grid). |
| `scripts/landmarks_final_build.py`, `scripts/landmarks_final_stats.py` | The two-observer consensus rules that produced the released landmark set (Supplementary S6). |
| `scripts/_inter_observer.py`, `scripts/_inter_agreement.py`, `scripts/_intra_*.py`, `scripts/_mr_duplicates.py`, `scripts/_lm_robustness.py` | Observer-variability analyses. |
| `scripts/_final_landmarks_rerun.sh`, `scripts/_final_lm_subject3_finish.sh`, `scripts/_extra_run_all.sh` | Runners of the complete re-scoring. |
| `paper_tools/build_numbers.py` | Every number of the paper as a LaTeX macro, from `tables/`; `build_paper.sh`, `build_journal.sh`, `build_zip.py`, `check_build.py` compile and check the manuscript. |
| `paper_tools/fig_*.py` | Figures 1 to 6. |
| `tables/` | `PER_CASE.csv` (every arm, every registration), `LANDMARKS.csv` (per landmark), `STEPS.csv`, `SENSITIVITY.csv`, `AFFINE_VERIFY.csv`, and the summaries of the other deformable methods. |

## Conventions

- Affines are 4x4 matrices in the NiftyReg convention (Affine * Ref = Flo), mapping iUS world coordinates to MR
  world coordinates. Displacement fields `disp_mm.nii.gz` are 3-vector images in world millimeters on the 1 mm
  working grid of the iUS; the complete transform is `A (x + u(x))`.
- Landmark files are 3D Slicer markups in LPS; the readers convert them to RAS. Pairs are matched by the integer
  that ends the label.
- Case-mean TRE is the mean over the landmark pairs of a case; cohort figures are means of case means.
