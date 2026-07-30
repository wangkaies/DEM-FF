# DEM-FF: Deep Earth Materials Force Field

A Universal Machine Learning Force Field for Earth’s Interior Studies

## Supported Elements (25)

H, He, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Ti, Mn, Fe, Ni, Kr, Xe, W

## Features

- Supports simulation of minerals, fluids, silicate melts, Fe-alloys (or possibly any system composed of supported elements) up to 360 GPa and 10,000+ K
- Long-range electrostatics via Latent Ewald Summation (LES)
- Electronic temperature dependence
- Free energy calculation via thermodynamic integration from the Uhlenbeck–Ford model (UFM) reference potential (p = 50, σ = 1.2 Å)

## Installation

### 1. Install MACE

```bash
pip install --upgrade pip
pip install mace-torch
```

(Optional but recommended) Install CUDA acceleration:

```bash
pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12
```

See: https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html

### 2. Install LES

```bash
git clone https://github.com/ChengUCB/les.git
cd les
pip install -e .
```

See: https://github.com/ChengUCB/les

### 3. Install ASE

```bash
pip install ase
```

See: https://wiki.fysik.dtu.dk/ase/

## Quick Start

### Simulation Example

1. Place your input structure (e.g., `POSCAR`) in the working directory.
2. Place the model file `DEMFF.model` in the same directory.
3. Edit the USER SETTINGS section in `ase_DEMFF_simulation.py`:

```python
TEMPERATURE = 4000.0    # K
PRESSURE    = 50.0      # GPa
TIMESTEP    = 1.0       # fs
TOTAL_STEPS = 100000
ENSEMBLE    = "npt_iso" # "nvt", "npt_iso", "npt_aniso", "nve", "optimize"
```

4. Run:

```bash
python ase_DEMFF_simulation.py
```

### Output Files

| File | Content |
|------|---------|
| `md.traj` | ASE trajectory (read with `ase.io.Trajectory`) |
| `md.log` | MD log (step, time, energy, temperature) |
| `output.dat` | Thermodynamic data with stress tensor |
| `POSCAR_final.vasp` | Final structure in VASP format |

### Continuation Runs

To continue a simulation from the last frame of a previous trajectory:

```python
INPUT_STRUCTURE = "md.traj"    # reads last frame automatically
INIT_VELOCITIES = False        # keep velocities from trajectory
APPEND          = True         # append to existing output files
```

## Electronic Temperature

DEM-FF uses electronic temperature as an input feature, which is set automatically from the simulation temperature:

```python
atoms.info["elec_temp"] = temperature_K * 8.61733e-5  # in eV
```

This is handled internally by `ase_DEMFF_simulation.py`. For custom workflows, set `elec_temp` in `atoms.info` and pass `info_keys={"elec_temp": "elec_temp"}` to `MACECalculator`.

## Using DEM-FF in Custom Scripts

```python
import torch
from ase.io import read
from mace.calculators import MACECalculator

atoms = read("POSCAR")
atoms.info["elec_temp"] = 4000.0 * 8.61733e-5  # eV

device = "cuda" if torch.cuda.is_available() else "cpu"
calc = MACECalculator(
    model_paths="DEMFF.model",
    default_dtype="float32",
    device=device,
    info_keys={"elec_temp": "elec_temp"},
)
atoms.calc = calc

energy = atoms.get_potential_energy()       # eV
forces = atoms.get_forces()                 # eV/Ang
stress = atoms.get_stress(voigt=False)      # eV/Ang^3 (3x3)
```

## Citation

If you use DEM-FF in your research, please cite:

> K. Wang, Y. Zhang, X. Lu. DEM-FF: A Universal Machine Learning Force Field for Earth's Interior Studies. DOI: [10.21203/rs.3.rs-9140615/v1](https://doi.org/10.21203/rs.3.rs-9140615/v1)
