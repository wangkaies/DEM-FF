# DEM-FF: Deep Earth Mantle Force Field

A MACE-LES (Long-range Equivariant Solver) machine learning interatomic potential for simulating deep Earth materials at extreme conditions.

## Supported Elements

H, He, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Ti, Mn, Fe, Ni, Kr, Xe, W

(25 elements, Z = 1, 2, 5–10, 11–20, 22, 25, 26, 28, 36, 54, 74)

## Features

- Energy, forces, stress, and virial predictions
- Long-range electrostatics via learned latent charges (LES)
- Electronic temperature dependence
- GPU-accelerated via CUDA (with optional cuequivariance support)

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

### Run a simulation

1. Place your input structure as `POSCAR` in the working directory.
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

### Available Ensembles

| Ensemble | Description | Typical Use |
|----------|-------------|-------------|
| `nvt` | Bussi stochastic velocity rescaling | Production runs at fixed volume |
| `npt_iso` | MTK isotropic NPT (Nosé-Hoover chains) | Equilibration and production of liquids/melts |
| `npt_aniso` | MTK full-cell NPT (Nosé-Hoover chains) | Crystals, two-phase simulations, melting point |
| `nve` | Velocity Verlet (microcanonical) | Transport properties, benchmarking |
| `optimize` | LBFGS cell + position optimization | Static structure relaxation |

### Recommended Timesteps

| System | Timestep |
|--------|----------|
| Metals (Fe, Ni, ...) | 1.0 fs |
| Silicates / oxides | 0.5–1.0 fs |
| Hydrous systems (H₂O) | 0.5 fs |

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

DEM-FF models support electronic temperature as an input feature. This is set automatically from the simulation temperature:

```python
atoms.info["elec_temp"] = temperature_K * 8.61733e-5  # in eV
```

This is handled internally by the simulation script. For custom workflows, ensure `elec_temp` is set in `atoms.info` and pass `info_keys={"elec_temp": "elec_temp"}` to `MACECalculator`.

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
forces = atoms.get_forces()                 # eV/Å
stress = atoms.get_stress(voigt=False)      # eV/Å³ (3×3)
```

## Typical Workflow

### Single-phase melt properties

```
1. Build initial structure (random or from experiment)
2. Melt at high T (NVT, 8000 K, ~50 ps)
3. Quench to target T (NVT, short)
4. NPT equilibration at target (T, P)
5. Average cell volume from last 60% of NPT
6. NVT or NVE production at averaged cell
7. Analyze trajectory (RDF, MSD, diffusion, species)
```

### Melting point (two-phase coexistence)

```
1. Build crystal supercell (elongated along one axis)
2. NPT equilibrate at trial T
3. Melt one half (fix bottom, NVT melt top at 8000 K)
4. Release constraints → NPT/NPH production
5. If crystal grows → T < Tm; if shrinks → T > Tm
6. Binary search on T
```

## Citation

If you use DEM-FF in your research, please cite:

[Citation information to be added]

## License

[License information to be added]
