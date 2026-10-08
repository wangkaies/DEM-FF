# Public regression record

Measured on 2026-09-12 with the upstream DEM-FF checkpoint SHA256
`f8f55225fda02adc4b768d71bc4dd4cadafb876b725c406b44934b9a63be91ca`.
Detailed sanitized results are in [validation-results.json](validation-results.json).

## Environment and execution

Python 3.12.13; DeePMD 3.2.0; Torch 2.10.0, CUDA 12.9; MACE 0.3.16;
e3nn 0.4.4; LES 0.2.0 (pinned revision in requirements); ASE 3.29.0;
one NVIDIA A100-SXM4-80GB, four CPU threads, float64.

The optional package was built as a wheel and installed into a fresh
`--system-site-packages` venv using `pip install --no-deps --no-build-isolation`.
The runtime import path was checked to point to that venv's installed package.
Dependencies were inherited from an existing validated installation. This does
not establish clean-machine dependency installation or official wheel portability.

Eight lightweight input/manifest tests passed on both the local machine and the
GPU host. Inference passed in the first run. That run then exposed an uninitialized
output-path variable in the newly prepared training regression script, before
training started. The path was corrected; the training regression was rerun and
completed with exit code 0. The final training check took approximately 75 seconds.
The passing inference and corrected training results are combined below; a single
uninterrupted post-fix shell orchestration run is not claimed.

## Results

| Check | Observed result |
|---|---|
| Manifest portability, overwrite refusal, checksum tamper, input failure cases | 8 tests pass |
| Author's 350-atom POSCAR, energy difference vs native MACE | 0 eV |
| Author POSCAR, maximum force difference vs native MACE | 7.11e-15 eV/angstrom |
| Author POSCAR, maximum total-virial finite-difference error | 1.07e-6 eV |
| Small triclinic fixture, sampled force finite difference | 6.33e-7 eV/angstrom |
| Small fixture, six independent strain components | maximum error 1.12e-8 eV |
| Rotation, permutation, periodic image, electronic-temperature batch | pass |
| LES contribution on small fixture | nonzero, approximately -1.19425 eV |
| Separate E/F/V losses | nonzero finite gradients in interactions, LES readouts, joint embedding |
| Selected LES parameter derivative for each loss | matches central finite differences within rtol 2e-3, atol 1e-6 |
| 12-step native training | all three weight groups change; optimizer state present |
| Same six-frame weighted loss before/after | 4.16667e-4 to 3.29308e-4 |
| Restart to step 16 and separate four-step native fine-tune | pass |
| Checkpoint reload, maximum E/F/V difference | below 3.35e-15 in their respective units |
| Export, maximum E/F/V difference | below 2.06e-15 in their respective units |
| Exported standard dp test on two synthetic validation frames | exit code 0; finite outputs |

The author's POSCAR native-stress-derived virial differs by up to 9.36574 eV
from the full energy derivative. It is deliberately not used as the parity
reference; the finite-difference checks above validate the adapter's strain
derivative. See the version-specific [design note](design.md).

## Reproduce

Follow the installation and trusted-checkpoint setup in [README](../README.md),
then run from the integration directory in a single-GPU allocation:

```bash
bash scripts/run_smoke.sh work/DEMFF.demff ../../POSCAR work/smoke
```

Individual stages, with new output paths:

```bash
python -m unittest discover -s tests -v
python scripts/validate_inference.py work/DEMFF.demff ../../POSCAR work/inference-only
python scripts/validate_training.py work/DEMFF.demff work/training-only
```

The generated training data contain O/H only, six atoms, six train and two
validation frames, and a constant +0.05 eV energy shift. Labels come from the
same model used to initialize training. Loss reduction, numerical parity, and
nonzero gradients establish software behavior only. They do not measure DFT
accuracy, extrapolation, physical electronic-temperature calibration, phosphorus
coverage, or MD stability.
