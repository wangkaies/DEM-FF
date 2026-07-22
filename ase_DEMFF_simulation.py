"""
ase_DEMFF_simulation.py — Run MD simulations with DEM-FF (MACE-LES) models.

Supports NVT, NPT (isotropic/anisotropic), NVE ensembles using pure ASE.
No external packages required beyond ASE, MACE, and LES.

Usage:
    python ase_DEMFF_simulation.py

Modify the USER SETTINGS section below to configure your simulation.
"""

import os
import numpy as np
import torch

from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.verlet import VelocityVerlet
from ase.md.bussi import Bussi
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md import IsotropicMTKNPT, MTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS

from mace.calculators import MACECalculator


# ============================================================
#  USER SETTINGS — modify this section for your simulation
# ============================================================

# Input structure (VASP POSCAR, CIF, XYZ, etc.)
INPUT_STRUCTURE = "POSCAR"

# Model path
MODEL_PATH = "./DEMFF.model"

# Simulation parameters
TEMPERATURE    = 4000.0   # K
PRESSURE       = 30.0      # GPa (used for NPT only)
TIMESTEP       = 1.0      # fs  (0.5 for silicates at high T, 1.0 for metals)
TOTAL_STEPS    = 100000   # total MD steps
LOG_INTERVAL   = 100      # logging interval (steps)

# Ensemble: "nvt", "npt_iso", "npt_aniso", "nve", "optimize"
ENSEMBLE = "nvt"

# Whether to initialize velocities (set False to continue from a previous run)
INIT_VELOCITIES = True

# Output file names
TRAJ_FILE   = "md.traj"
LOG_FILE    = "md.log"
OUTPUT_FILE = "output.dat"
FINAL_FILE  = "POSCAR_final.vasp"

# Append to existing trajectory? (for continuation runs)
APPEND = False


# ============================================================
#  SETUP — normally no need to modify below
# ============================================================

def setup_calculator(model_path):
    """Create MACECalculator with automatic CPU/GPU selection."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    calc = MACECalculator(model_paths=model_path, default_dtype="float32", device=device, info_keys={"elec_temp": "elec_temp"},) #enable_cueq=True This is for acceleration on GPU
    return calc


def setup_atoms(input_structure, temperature, calculator):
    """Read structure, set electronic temperature, attach calculator."""
    try:
        atoms = read(input_structure)
    except (AssertionError, Exception):
        # Some VASP files have trailing velocity blocks that cause parsing errors.
        # Fall back to reading only the coordinate section.
        import io as _io
        lines = open(input_structure).read().splitlines()
        cursor = 7
        if lines[cursor].strip().lower().startswith("s"):
            cursor += 1
        cursor += 1
        natoms = sum(int(x) for x in lines[6].split())
        trimmed = "\n".join(lines[:cursor + natoms]) + "\n"
        atoms = read(_io.StringIO(trimmed), format="vasp")
    atoms.info["elec_temp"] = temperature * 8.61733e-5  # eV
    atoms.calc = calculator
    print(f"Structure: {input_structure}")
    print(f"  Composition: {atoms.get_chemical_formula()}")
    print(f"  Atoms: {len(atoms)}")
    print(f"  Cell volume: {atoms.get_volume():.1f} A^3")
    return atoms


def init_velocities(atoms, temperature):
    """Initialize Maxwell-Boltzmann velocities."""
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
    Stationary(atoms)
    ZeroRotation(atoms)


# ============================================================
#  ENSEMBLE FUNCTIONS
# ============================================================

def run_nvt(atoms, temperature, timestep, steps, log_interval,
            traj_file, log_file, output_file, append=False):
    """NVT ensemble using Bussi stochastic velocity rescaling.
    Bussi et al., J. Chem. Phys. 126, 014101 (2007)."""
    mode = "a" if append else "w"
    dyn = Bussi(atoms, timestep * units.fs, temperature_K=temperature, taut=100 * units.fs, trajectory=traj_file, logfile=log_file, loginterval=log_interval, append_trajectory=append)
    dyn.attach(MDLogger(dyn, atoms, output_file,header=(not append), stress=True,peratom=False, mode=mode),interval=log_interval)
    print(f"\nRunning NVT (Bussi): {temperature} K, {steps} steps, dt = {timestep} fs")
    dyn.run(steps)
    return dyn


def run_npt_iso(atoms, temperature, pressure, timestep, steps, log_interval,
                traj_file, log_file, output_file, append=False):
    """Isotropic NPT using Martyna-Tobias-Klein (Nose-Hoover chains).
    Martyna et al., J. Chem. Phys. 101, 4177 (1994)."""
    mode = "a" if append else "w"
    dyn = IsotropicMTKNPT(atoms, timestep * units.fs,temperature_K=temperature,pressure_au=pressure * units.GPa,tdamp=100 * units.fs,pdamp=1000 * units.fs, trajectory=traj_file,logfile=log_file,loginterval=log_interval, append_trajectory=append)
    dyn.attach(MDLogger(dyn, atoms, output_file, header=(not append), stress=True,peratom=False, mode=mode),interval=log_interval)
    print(f"\nRunning NPT (iso, MTK): {temperature} K, {pressure} GPa, {steps} steps")
    dyn.run(steps)
    return dyn


def run_npt_aniso(atoms, temperature, pressure, timestep, steps, log_interval,
                  traj_file, log_file, output_file, append=False):
    """Anisotropic NPT using Martyna-Tobias-Klein (full cell fluctuations).
    Suitable for crystalline phases and two-phase simulations."""
    mode = "a" if append else "w"
    dyn = MTKNPT(atoms, timestep * units.fs,temperature_K=temperature,pressure_au=pressure * units.GPa,tdamp=100 * units.fs,pdamp=1000 * units.fs,trajectory=traj_file,logfile=log_file,loginterval=log_interval,append_trajectory=append)
    dyn.attach(MDLogger(dyn, atoms, output_file,header=(not append), stress=True,peratom=False, mode=mode),interval=log_interval)
    print(f"\nRunning NPT (aniso, MTK): {temperature} K, {pressure} GPa, {steps} steps")
    dyn.run(steps)
    return dyn


def run_nve(atoms, temperature, timestep, steps, log_interval,
            traj_file, log_file, output_file, append=False):
    """NVE ensemble using Velocity Verlet."""
    mode = "a" if append else "w"
    dyn = VelocityVerlet(atoms, timestep * units.fs,trajectory=traj_file,logfile=log_file, loginterval=log_interval,  append_trajectory=append)
    dyn.attach(MDLogger(dyn, atoms, output_file, header=(not append), stress=True, peratom=False, mode=mode), interval=log_interval)
    print(f"\nRunning NVE: {steps} steps, dt = {timestep} fs")
    dyn.run(steps)
    return dyn


def run_optimize(atoms, pressure=0.0, fmax=0.01, max_steps=500):
    """Cell + position optimization at target pressure."""
    target_p_ev = pressure / 160.2176634
    ecf = FrechetCellFilter(atoms, scalar_pressure=target_p_ev)
    opt = LBFGS(ecf, logfile="opt.log")
    print(f"\nOptimizing structure to {pressure} GPa (fmax = {fmax} eV/A)")
    opt.run(fmax=fmax, steps=max_steps)
    stress = atoms.get_stress()
    P_final = -np.mean(stress[:3]) * 160.2176634
    print(f"  Converged in {opt.nsteps} steps, P = {P_final:.3f} GPa")
    return opt


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    calc = setup_calculator(MODEL_PATH)
    atoms = setup_atoms(INPUT_STRUCTURE, TEMPERATURE, calc)

    if ENSEMBLE == "optimize":
        run_optimize(atoms, pressure=PRESSURE)
    else:
        if INIT_VELOCITIES:
            init_velocities(atoms, TEMPERATURE)

        if ENSEMBLE == "nvt":
            run_nvt(atoms, TEMPERATURE, TIMESTEP, TOTAL_STEPS, LOG_INTERVAL,TRAJ_FILE, LOG_FILE, OUTPUT_FILE, APPEND)

        elif ENSEMBLE == "npt_iso":
            run_npt_iso(atoms, TEMPERATURE, PRESSURE, TIMESTEP, TOTAL_STEPS,LOG_INTERVAL, TRAJ_FILE, LOG_FILE, OUTPUT_FILE, APPEND)

        elif ENSEMBLE == "npt_aniso":
            run_npt_aniso(atoms, TEMPERATURE, PRESSURE, TIMESTEP, TOTAL_STEPS,LOG_INTERVAL, TRAJ_FILE, LOG_FILE, OUTPUT_FILE, APPEND)

        elif ENSEMBLE == "nve":
            run_nve(atoms, TEMPERATURE, TIMESTEP, TOTAL_STEPS, LOG_INTERVAL,TRAJ_FILE, LOG_FILE, OUTPUT_FILE, APPEND)

        else:
            raise ValueError(f"Unknown ensemble: {ENSEMBLE}")

    atoms.wrap()
    write(FINAL_FILE, atoms, format="vasp", vasp5=True, direct=True)
    print(f"\nDone. Final structure saved to {FINAL_FILE}")
