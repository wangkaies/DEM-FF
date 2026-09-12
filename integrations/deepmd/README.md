# Optional DeePMD integration for DEM-FF

Experimental Python inference and native DeePMD PyTorch training for the complete
DEM-FF `MACELES` checkpoint, including LES and electronic-temperature conditioning.
This is an optional package. The repository's ASE simulation workflow remains the
reference for its existing MD examples.

The adapter uses DeePMD's data loader, losses, optimizer, checkpoint, restart, and
fine-tune commands. It does not convert DEM-FF into a standard Deep Potential
architecture. General MACE integration already exists in
[DeePMD-GNN](https://github.com/deepmodeling/deepmd-gnn); see
[scope and related work](docs/design.md).

## Install in a separate environment

Commands below assume Linux/bash and a local checkout of this repository.
From the repository root:

```bash
conda create -n demff-deepmd python=3.12 pip -y
conda activate demff-deepmd
cd integrations/deepmd
```

Install PyTorch **2.10.0** using the command appropriate for your driver and
platform from the [official version table](https://pytorch.org/get-started/previous-versions/).
For example, the official CUDA 12.8 wheel is:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-validated.txt
python -m pip install --no-deps -e .
```

The regression was run with Python 3.12.13, DeePMD 3.2.0, MACE 0.3.16,
Torch 2.10.0 (site-provided CUDA 12.9 build), e3nn 0.4.4, ASE 3.29.0, and
LES 0.2.0 at the revision pinned in `requirements-validated.txt`.
The optional package was freshly built and installed in a separate venv sharing
that preinstalled dependency stack. A clean conda bootstrap and the official
CUDA 12.8 wheel combination have not been validated here. MPI runtime selection
is installation-specific; use a working DeePMD Python environment, rather than
mixing unrelated MPI libraries.

This pinned MACE/e3nn stack loads Python checkpoint objects. Use only a trusted
checkpoint. In this activated test environment, enable the legacy loader required
by these pinned versions:

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4
export DP_INTRA_OP_PARALLELISM_THREADS=4
export DP_INTER_OP_PARALLELISM_THREADS=1
python scripts/check_environment.py
mkdir -p work
demff-deepmd register ../../DEMFF.model work/DEMFF.demff
```

Registration writes a relative-path JSON manifest with the weight's SHA256;
it does not copy, convert, or change the original model. Keep the model at its
referenced path, or move it together with the manifest while preserving that
relative layout. A `.demff` file alone does not contain any weights.

## Run the public regression

Inside a single-GPU allocation, from `integrations/deepmd`:

```bash
python -m unittest discover -s tests -v
bash scripts/run_smoke.sh work/DEMFF.demff ../../POSCAR work/smoke
```

Use a new output directory for every run. This script does not submit a job.
It checks the author's existing POSCAR, energy/force agreement, finite-difference
force and virial, symmetry and temperature responses, independent energy/force/
virial training gradients, weight updates, checkpoint restart, native fine-tune,
export, and `dp test`. See [validation details](docs/validation.md).

Training fixtures are generated locally from the model: six perturbed water
atoms, six training frames, two validation frames, and a constant +0.05 eV
energy shift. They are **synthetic software fixtures, not DFT data**. The training
fixture exercises O/H, including an unused P entry in its type map; it is not a
P-containing accuracy benchmark. No model or research training set is bundled.

## Train on your own DeePMD data

Prepare distinct `deepmd/npy` systems with the same explicit `type_map.raw` order.
Every `set.*` must contain these arrays:

| File | Shape for F frames and N atoms | Units / meaning |
|---|---|---|
| `coord.npy` | `(F, 3N)` | angstrom |
| `box.npy` | `(F, 9)` | row-major cell vectors, angstrom |
| `energy.npy` | `(F,)` or `(F, 1)` | total energy, eV |
| `force.npy` | `(F, 3N)` | eV/angstrom |
| `virial.npy` | `(F, 9)` | total virial, eV; not stress |
| `fparam.npy` | `(F, 1)` | electronic kBT, eV |

Each system also needs `type.raw` and `type_map.raw`. See the
[DeePMD 3.2 format specification](https://docs.deepmodeling.com/projects/deepmd/en/v3.2.0/data/data-conv.html).
For the ASE stress convention, virial is `-volume * stress`; verify the convention
when converting a different code's output. This adapter requires full right-handed
3D periodic cells. The preparation helper requires E/F/V labels in every system.

**Choose label and electronic-temperature conventions explicitly.** Physical
electronic temperature T maps to `kBT = 8.61733e-5 * T` eV. A Gaussian DFT smearing
width is not automatically this temperature. For a fixed-smearing dataset, a fixed
conditioning value is a modeling choice to document and validate. The helper
never invents `fparam`, shifts labels, or changes energy/free-energy conventions.

Replace the data paths below with your own prepared systems:

```bash
python scripts/prepare_input.py \
  --foundation work/DEMFF.demff \
  --train /path/to/train/system1 /path/to/train/system2 \
  --valid /path/to/validation/system1 \
  --steps 1000 --lr 1e-5 --output work/finetune/input.json
cd work/finetune
python -m deepmd --pt train input.json
demff-deepmd export model.ckpt.pt adapted.demff
python -m deepmd test -m adapted.demff -s /path/to/heldout/system -n 100 -d heldout
cd ../..
```

The original DEM-FF weights are loaded through `model.foundation`; do not pass
the original `.model` to DeePMD's `--finetune`. To continue an existing **native
DeePMD `.pt` checkpoint**, use:

```bash
# In the run directory, after raising numb_steps in input.json:
python -m deepmd --pt train input.json --restart model.ckpt.pt
# For a separate downstream run with the same type_map:
python -m deepmd --pt train next-input.json --finetune /path/to/model.ckpt.pt
```

Restart resumes the training state; fine-tune starts a new optimization run.
Keep the original foundation manifest and weights available: construction during
restart/export currently still uses the saved `foundation` path. Export produces
an `adapted.model` MACE-LES object plus `adapted.demff`; neither existing output is
overwritten. Use a new output basename for a second export.

The preparation helper checks shapes, finite values, cells, type-map order,
and matching ordered geometries between train and validation (quantized to
1e-5 angstrom). This diagnostic does not establish independence under permutations,
translations, periodic wrapping, or trajectory correlations. Split trajectories
and assess held-out accuracy separately. Default loss weights and step count are
engineering starting values, not tuned scientific recommendations. Original atomic
reference energies are retained; no automatic energy-offset fit is performed.

## Python inference

```python
import numpy as np
from deepmd.infer import DeepPot

dp = DeepPot("work/DEMFF.demff", device="cuda", default_dtype="float64")
symbols = ["O", "H", "H"]
types = [dp.get_type_map().index(s) for s in symbols]
coord = np.array([[1., 1., 1., 1.96, 1., 1., .76, 1.93, 1.]])
box = (np.eye(3) * 7).reshape(1, 9)
energy, force, virial = dp.eval(coord, box, types, fparam=np.array([[.1723466]]))
```

## Current limits

- Complete-system Python evaluation; graphs are rebuilt per frame.
- Float64 training, one process / one GPU validated. CPU inference is available;
  this public regression uses CUDA. Distributed training is unvalidated.
- No `dp freeze`, standard DeePMD TorchScript export, C++/LAMMPS pair style,
  MPI domain decomposition, or external neighbor-list integration.
- No atomic energy/virial, Hessian, atomic parameters, spin, or virtual atoms.
- Fine-tuning an existing native checkpoint requires the same type-map order.
- Uses pinned private MACE and DeePMD interfaces; other versions need regression.
- Export parity is a software check, not proof of MD stability or physical accuracy.

For the pinned MACE/LES stack, virial is computed by differentiating the complete
energy under cell/coordinate strain, then checked by finite differences. Native
MACE stress is not used as the reference; see [the implementation note](docs/design.md).
