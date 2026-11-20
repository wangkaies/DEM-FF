"""
ase_DEMFF_simulation.py — Run MD simulations with DEM-FF (MACE-LES) models.

Supports NVT, NPT (isotropic / anisotropic orthorhombic / triclinic), NVE,
and cell+position optimization using pure ASE, MACE, and LES.

Usage:
    1. Modify the USER SETTINGS section below (TEMPERATURE, PRESSURE, etc.).
    2. Run:
           python ase_DEMFF_simulation.py

(Optional: command-line flags like --T 4800 --P 50 --steps 500 can also override USER SETTINGS if needed.)
"""

import argparse
import io as _io
import os
import sys
import time
import numpy as np
import torch

# Fallback to bundled `les_pkg/src` if `les` is not yet pip-installed
_LES_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "les_pkg", "src")
if os.path.isdir(_LES_SRC) and _LES_SRC not in sys.path:
    sys.path.append(_LES_SRC)

from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.verlet import VelocityVerlet
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
try:
    from ase.md.velocitydistribution import thermalize_momenta
except ImportError:
    thermalize_momenta = None
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS

# Compatible imports across ASE versions (e.g. ASE 3.25 - 3.27+)
try:
    from ase.md.nose_hoover_chain import IsotropicMTKNPT, MTKNPT
except ImportError:
    from ase.md import IsotropicMTKNPT, MTKNPT

try:
    from ase.md.melchionna import MelchionnaNPT
except ImportError:
    from ase.md.npt import NPT as MelchionnaNPT

from mace.calculators import MACECalculator


# ============================================================
#  USER SETTINGS — modify this section for your simulation
# ============================================================

# Input structure (VASP POSCAR, Extended XYZ, CIF, ASE traj, etc.)
# Built-in 240 GPa relaxed HCP Fe structures:
#   "POSCAR_hcp_Fe.vasp"     (180-atom orthorhombic cell)
#   "POSCAR_hcp_Fe_hex.vasp" (200-atom hexagonal cell, gamma = 120 deg)
INPUT_STRUCTURE = "POSCAR_hcp_Fe.vasp"

# Model path
MODEL_PATH = "./DEMFF.model"

# Simulation parameters
TEMPERATURE  = 5000.0   # K   (at 240 GPa: ~5000 K solid, ~8000 K melts clearly)
PRESSURE     = 240.0    # GPa (used for NPT and optimize)
TIMESTEP     = 2.0      # fs  (1.0-2.0 for Fe/metals; 0.5 for silicates/H at high T)
TOTAL_STEPS  = 500      # total MD steps (500 steps @ 2 fs = 1.0 ps)
LOG_INTERVAL = 10       # output interval (steps) for trajectory & thermo log

# Ensemble options:
#   "nvt"       : Constant N, V, T (Bussi thermostat)
#   "npt_iso"   : Constant N, P, T — isotropic scaling (recommended for melting/liquids)
#   "npt_aniso" : Constant N, P, T — orthorhombic a, b, c independent, angles fixed
#   "npt_tri"   : Constant N, P, T — full triclinic (a, b, c and alpha, beta, gamma all free)
#   "nve"       : Constant N, V, E (Velocity Verlet)
#   "optimize"  : Cell + atomic position relaxation at target PRESSURE
ENSEMBLE = "npt_iso"

# Barostat / thermostat damping constants (fs)
TDAMP = 100.0           # thermostat time constant (fs)
PDAMP = 500.0           # barostat time constant (fs)
BULK_MODULUS_GPA = 1000.0  # estimated bulk modulus (GPa), used only by npt_aniso

# Whether to initialize velocities (set False to continue from a previous run)
INIT_VELOCITIES = True

# Output file names (default trajectory format: Extended XYZ for OVITO/VESTA/ASE GUI)
TRAJ_FILE   = "md.xyz"
OUTPUT_FILE = "output.dat"
FINAL_FILE  = "POSCAR_final.vasp"

# Append to existing trajectory & log? (for continuation runs)
APPEND = False


# ============================================================
#  SETUP — normally no need to modify below
# ============================================================

EV_A3_TO_GPA = 1.0 / units.GPa  # 160.2176634 GPa per (eV/A^3)
AVOGADRO = 6.02214076e23


def setup_calculator(model_path):
    """Create MACECalculator with automatic CPU/GPU selection."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    calc = MACECalculator(
        model_paths=model_path,
        default_dtype="float32",
        device=device,
        info_keys={"elec_temp": "elec_temp"},
        # enable_cueq=True,  # Uncomment for CUDA cuEquivariance acceleration on GPU
    )
    return calc


def setup_atoms(input_structure, temperature, calculator):
    """Read structure (last frame if multi-frame), set elec_temp, attach calculator."""
    try:
        atoms = read(input_structure, index=-1)
    except Exception:
        # Some VASP files have trailing velocity blocks that cause parsing errors.
        lines = open(input_structure, encoding="utf-8", errors="ignore").read().splitlines()
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
    print(f"  Composition : {atoms.get_chemical_formula()}")
    print(f"  Atoms       : {len(atoms)}")
    lengths = atoms.cell.lengths()
    angles = atoms.cell.angles()
    print(
        f"  Cell        : a={lengths[0]:.3f}, b={lengths[1]:.3f}, c={lengths[2]:.3f} A "
        f"(alpha={angles[0]:.1f}, beta={angles[1]:.1f}, gamma={angles[2]:.1f} deg)"
    )
    print(f"  Volume      : {atoms.get_volume():.2f} A^3")
    return atoms


def init_velocities(atoms, temperature):
    """Initialize Maxwell-Boltzmann velocities with zero net momentum."""
    if thermalize_momenta is not None:
        thermalize_momenta(atoms, temperature_K=temperature, exact_temperature=True)
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
    Stationary(atoms)
    if not any(atoms.pbc):
        ZeroRotation(atoms)


def ensure_triangular_cell(atoms):
    """Rotate cell into lower-triangular standard form if required (for MelchionnaNPT)."""
    cell = np.array(atoms.get_cell())
    is_upper = np.allclose(np.tril(cell, -1), 0.0, atol=1e-10)
    is_lower = np.allclose(np.triu(cell, 1), 0.0, atol=1e-10)
    if is_upper or is_lower:
        return
    momenta = atoms.get_momenta().copy()
    rcell, Q = atoms.cell.standard_form()
    atoms.set_cell(rcell, scale_atoms=True)
    if np.any(momenta != 0.0):
        atoms.set_momenta(momenta @ Q.T)


# ============================================================
#  UNIFIED TRAJECTORY & THERMODYNAMIC LOGGER
# ============================================================

class SimulationLogger:
    """Writes Extended XYZ (or .traj) trajectory, full thermo log, and live table."""

    FILE_HEADER = (
        f"# {'Step':>7s} {'Time(ps)':>9s} {'Epot(eV)':>13s} {'Ekin(eV)':>11s} "
        f"{'Etot(eV)':>13s} {'T(K)':>8s} {'P(GPa)':>8s} "
        f"{'Pxx(GPa)':>9s} {'Pyy(GPa)':>9s} {'Pzz(GPa)':>9s} "
        f"{'Pyz(GPa)':>9s} {'Pxz(GPa)':>9s} {'Pxy(GPa)':>9s} "
        f"{'Vol(A^3)':>10s} {'Rho(g/cm3)':>10s} "
        f"{'a(A)':>8s} {'b(A)':>8s} {'c(A)':>8s} "
        f"{'alpha':>7s} {'beta':>7s} {'gamma':>7s} "
        f"{'MSD(A^2)':>9s} {'dt_wall(s)':>10s}\n"
    )

    SCREEN_HEADER = (
        f"{'Step':>6s} | {'Time(ps)':>8s} | {'Epot(eV)':>11s} | {'Ekin(eV)':>9s} | "
        f"{'T(K)':>7s} | {'P(GPa)':>7s} {'[Pxx,Pyy,Pzz]':>23s} | "
        f"{'Cell a,b,c(A)':>19s} | {'MSD(A^2)':>8s} | {'Wall(s)':>7s}"
    )

    def __init__(self, dyn, atoms, timestep_fs, traj_file, output_file, append=False):
        self.dyn = dyn
        self.atoms = atoms
        self.timestep_fs = timestep_fs
        self.traj_file = traj_file
        self.output_file = output_file
        self.append = append
        self.step_offset = 0
        self.skip_first = False

        # Resume step count if appending to an existing output.dat
        if append and os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    data_lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
                if data_lines:
                    self.step_offset = int(data_lines[-1].split()[0])
                    self.skip_first = True
            except Exception:
                pass

        # Trajectory writer setup
        self.is_ase_traj = traj_file.lower().endswith(".traj")
        if self.is_ase_traj:
            self.ase_traj = Trajectory(traj_file, "a" if append else "w", atoms)
        else:
            self.ase_traj = None
            if (not append) and os.path.exists(traj_file):
                os.remove(traj_file)

        # Open thermo log file
        mode = "a" if (append and os.path.exists(output_file)) else "w"
        self.fout = open(output_file, mode, encoding="utf-8")
        if mode == "w":
            self.fout.write(self.FILE_HEADER)
            self.fout.flush()

        # Unwrapped coordinate tracking for accurate MSD across PBC
        self.pos0 = atoms.get_positions().copy()
        self.prev_scaled = atoms.get_scaled_positions(wrap=False).copy()
        self.unwrapped = atoms.get_positions().copy()

        # Timing & summary statistics
        self.t_start = time.perf_counter()
        self.t_last = self.t_start
        self.history = []

        print("\n" + self.SCREEN_HEADER)
        print("-" * len(self.SCREEN_HEADER))

    def _update_msd(self):
        scaled = self.atoms.get_scaled_positions(wrap=False)
        ds = scaled - self.prev_scaled
        ds -= np.round(ds)
        self.unwrapped += ds @ np.array(self.atoms.get_cell())
        self.prev_scaled = scaled.copy()
        disp = self.unwrapped - self.pos0
        # Subtract center-of-mass drift for cleaner MSD
        disp -= np.mean(disp, axis=0, keepdims=True)
        return float(np.mean(np.sum(disp ** 2, axis=1)))

    def __call__(self):
        raw_step = self.dyn.get_number_of_steps()
        if self.skip_first and raw_step == 0:
            self.skip_first = False
            self.t_last = time.perf_counter()
            return

        now = time.perf_counter()
        dt_wall = now - self.t_last
        self.t_last = now

        step = self.step_offset + raw_step
        time_ps = step * self.timestep_fs / 1000.0

        epot = float(self.atoms.get_potential_energy())
        ekin = float(self.atoms.get_kinetic_energy())
        etot = epot + ekin
        temp = float(self.atoms.get_temperature())

        # Thermodynamic pressure tensor P_ab = -sigma_ab (includes kinetic ideal-gas term)
        # ASE Voigt order: [xx, yy, zz, yz, xz, xy]
        stress_voigt = self.atoms.get_stress(voigt=True, include_ideal_gas=True)
        p_comp = -stress_voigt * EV_A3_TO_GPA
        pxx, pyy, pzz, pyz, pxz, pxy = [float(x) for x in p_comp]
        p_mean = (pxx + pyy + pzz) / 3.0

        vol = float(self.atoms.get_volume())
        mass_g = float(np.sum(self.atoms.get_masses())) / AVOGADRO
        rho = mass_g / (vol * 1e-24)  # g/cm^3

        a, b, c = [float(x) for x in self.atoms.cell.lengths()]
        alpha, beta, gamma = [float(x) for x in self.atoms.cell.angles()]

        msd = self._update_msd()

        # 1. Write trajectory frame
        if self.is_ase_traj:
            self.ase_traj.write()
        else:
            # Copy without triggering recalculation, wrap atoms into box for clean visualization
            frame = self.atoms.copy()
            frame.wrap()
            frame.info["step"] = step
            frame.info["time_ps"] = round(time_ps, 6)
            frame.info["energy_eV"] = epot
            frame.info["temperature_K"] = temp
            frame.info["pressure_GPa"] = p_mean
            write(self.traj_file, frame, format="extxyz", append=True)

        # 2. Write thermodynamic data line
        self.fout.write(
            f"  {step:7d} {time_ps:9.4f} {epot:13.5f} {ekin:11.5f} "
            f"{etot:13.5f} {temp:8.1f} {p_mean:8.2f} "
            f"{pxx:9.2f} {pyy:9.2f} {pzz:9.2f} "
            f"{pyz:9.2f} {pxz:9.2f} {pxy:9.2f} "
            f"{vol:10.3f} {rho:10.4f} "
            f"{a:8.4f} {b:8.4f} {c:8.4f} "
            f"{alpha:7.2f} {beta:7.2f} {gamma:7.2f} "
            f"{msd:9.4f} {dt_wall:10.3f}\n"
        )
        self.fout.flush()

        # 3. Print live summary line
        print(
            f"{step:6d} | {time_ps:8.3f} | {epot:11.3f} | {ekin:9.3f} | "
            f"{temp:7.1f} | {p_mean:7.2f} [{pxx:6.1f},{pyy:6.1f},{pzz:6.1f}] | "
            f"{a:5.2f},{b:5.2f},{c:5.2f} | {msd:8.3f} | {dt_wall:7.2f}",
            flush=True,
        )

        self.history.append((step, time_ps, epot, temp, p_mean, rho, msd))

    def close(self):
        if self.ase_traj is not None:
            self.ase_traj.close()
        self.fout.close()
        total_wall = time.perf_counter() - self.t_start
        print("-" * len(self.SCREEN_HEADER))
        print(f"Total wall time: {total_wall:.1f} s ({total_wall / 60.0:.2f} min)")
        if len(self.history) >= 2:
            arr = np.array(self.history)
            half = len(arr) // 2
            t_avg = np.mean(arr[half:, 3])
            p_avg = np.mean(arr[half:, 4])
            rho_avg = np.mean(arr[half:, 5])
            msd_end = arr[-1, 6]
            msd_half = arr[half, 6]
            dt_ps = arr[-1, 1] - arr[half, 1]
            # Diffusion coefficient D = d(MSD)/dt / 6 (in A^2/ps -> 1e-8 m^2/s)
            d_a2_ps = (msd_end - msd_half) / (6.0 * dt_ps) if dt_ps > 0 else 0.0
            print(
                f"Second-half averages: T = {t_avg:.1f} K, P = {p_avg:.2f} GPa, "
                f"Rho = {rho_avg:.3f} g/cm^3"
            )
            print(
                f"Final MSD = {msd_end:.3f} A^2, estimated D = {d_a2_ps:.3f} A^2/ps "
                f"({d_a2_ps * 1e-8:.2e} m^2/s)"
            )
            if msd_end > 2.0 and d_a2_ps > 0.1:
                print(">>> Quick melting check: LIQUID / MELTED (原子显著扩散，已融化)")
            elif msd_end < 0.8:
                print(">>> Quick melting check: SOLID / CRYSTAL (原子围绕格点振动，未融化)")
            else:
                print(">>> Quick melting check: TRANSITIONAL / PARTIALLY MELTED (接近熔点，建议增加步数观察)")


# ============================================================
#  ENSEMBLE FUNCTIONS
# ============================================================

def _run_with_logger(dyn, atoms, timestep, steps, log_interval, traj_file, output_file, append):
    logger = SimulationLogger(
        dyn, atoms, timestep, traj_file, output_file, append=append
    )
    dyn.attach(logger, interval=log_interval)
    try:
        dyn.run(steps)
    finally:
        logger.close()
    return dyn


def run_nvt(atoms, temperature, timestep, steps, log_interval,
            traj_file, output_file, append=False, tdamp=100.0):
    """NVT ensemble using Bussi stochastic velocity rescaling."""
    dyn = Bussi(
        atoms,
        timestep * units.fs,
        temperature_K=temperature,
        taut=tdamp * units.fs,
    )
    print(f"\nRunning NVT (Bussi): T = {temperature} K, {steps} steps, dt = {timestep} fs")
    return _run_with_logger(dyn, atoms, timestep, steps, log_interval, traj_file, output_file, append)


def run_npt_iso(atoms, temperature, pressure, timestep, steps, log_interval,
                traj_file, output_file, append=False, tdamp=100.0, pdamp=500.0):
    """Isotropic NPT using Martyna-Tobias-Klein (Nose-Hoover chains).
    Volume scales uniformly; cell shape and angles stay fixed.
    Recommended for liquids and melting simulations."""
    dyn = IsotropicMTKNPT(
        atoms,
        timestep * units.fs,
        temperature_K=temperature,
        pressure_au=pressure * units.GPa,
        tdamp=tdamp * units.fs,
        pdamp=pdamp * units.fs,
    )
    print(
        f"\nRunning NPT-iso (IsotropicMTKNPT): T = {temperature} K, P = {pressure} GPa, "
        f"{steps} steps, dt = {timestep} fs"
    )
    return _run_with_logger(dyn, atoms, timestep, steps, log_interval, traj_file, output_file, append)


def run_npt_aniso(atoms, temperature, pressure, timestep, steps, log_interval,
                  traj_file, output_file, append=False, tdamp=100.0, pdamp=500.0,
                  bulk_modulus_gpa=200.0):
    """Orthorhombic anisotropic NPT using MelchionnaNPT with diagonal strain mask.
    Box lengths (a, b, c) vary independently while angles remain fixed (no box shear).
    Suitable for anisotropic crystals (e.g. HCP c/a relaxation) and two-phase cells."""
    ensure_triangular_cell(atoms)
    pfactor = (pdamp * units.fs) ** 2 * (bulk_modulus_gpa * units.GPa)
    dyn = MelchionnaNPT(
        atoms,
        timestep * units.fs,
        temperature_K=temperature,
        externalstress=pressure * units.GPa,
        ttime=tdamp * units.fs,
        pfactor=pfactor,
        mask=np.eye(3, dtype=int),
    )
    print(
        f"\nRunning NPT-aniso (MelchionnaNPT, orthorhombic mask): T = {temperature} K, "
        f"P = {pressure} GPa, {steps} steps, dt = {timestep} fs"
    )
    return _run_with_logger(dyn, atoms, timestep, steps, log_interval, traj_file, output_file, append)


def run_npt_tri(atoms, temperature, pressure, timestep, steps, log_interval,
                traj_file, output_file, append=False, tdamp=100.0, pdamp=500.0):
    """Fully flexible triclinic NPT using Martyna-Tobias-Klein (MTKNPT).
    Full 3x3 cell fluctuations: lengths (a, b, c) AND angles (alpha, beta, gamma)
    all evolve freely. Best for solid-state phase transitions & triclinic crystals.
    Note: avoid long liquid runs in npt_tri as liquids have zero shear modulus."""
    dyn = MTKNPT(
        atoms,
        timestep * units.fs,
        temperature_K=temperature,
        pressure_au=pressure * units.GPa,
        tdamp=tdamp * units.fs,
        pdamp=pdamp * units.fs,
    )
    print(
        f"\nRunning NPT-tri (MTKNPT, full triclinic): T = {temperature} K, "
        f"P = {pressure} GPa, {steps} steps, dt = {timestep} fs"
    )
    return _run_with_logger(dyn, atoms, timestep, steps, log_interval, traj_file, output_file, append)


def run_nve(atoms, timestep, steps, log_interval, traj_file, output_file, append=False):
    """NVE ensemble using Velocity Verlet."""
    dyn = VelocityVerlet(atoms, timestep * units.fs)
    print(f"\nRunning NVE (VelocityVerlet): {steps} steps, dt = {timestep} fs")
    return _run_with_logger(dyn, atoms, timestep, steps, log_interval, traj_file, output_file, append)


def run_optimize(atoms, pressure=0.0, fmax=0.01, max_steps=500):
    """Cell + atomic position optimization at target pressure."""
    target_p_ev = pressure * units.GPa
    ecf = FrechetCellFilter(atoms, scalar_pressure=target_p_ev)
    opt = LBFGS(ecf, logfile="opt.log")
    print(f"\nOptimizing structure to {pressure} GPa (fmax = {fmax} eV/A)")
    opt.run(fmax=fmax, steps=max_steps)
    stress = atoms.get_stress()
    p_final = -np.mean(stress[:3]) * EV_A3_TO_GPA
    print(f"  Converged in {opt.nsteps} steps, P = {p_final:.3f} GPa")
    return opt


# ============================================================
#  CLI & MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DEM-FF molecular dynamics simulations with ASE."
    )
    parser.add_argument("--input", default=INPUT_STRUCTURE, help="Input structure file")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to DEMFF.model")
    parser.add_argument("--T", type=float, default=TEMPERATURE, help="Temperature (K)")
    parser.add_argument("--P", type=float, default=PRESSURE, help="Pressure (GPa)")
    parser.add_argument("--dt", type=float, default=TIMESTEP, help="Timestep (fs)")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS, help="Total MD steps")
    parser.add_argument("--interval", type=int, default=LOG_INTERVAL, help="Log/traj interval (steps)")
    parser.add_argument(
        "--ensemble",
        default=ENSEMBLE,
        choices=["nvt", "npt_iso", "npt_aniso", "npt_tri", "nve", "optimize"],
        help="Simulation ensemble",
    )
    parser.add_argument("--traj", default=TRAJ_FILE, help="Trajectory file (.xyz or .traj)")
    parser.add_argument("--out", default=OUTPUT_FILE, help="Thermodynamic log file")
    parser.add_argument("--final", default=FINAL_FILE, help="Final structure file (VASP)")
    parser.add_argument(
        "--continue",
        dest="cont",
        action="store_true",
        help="Continue from previous run (keep velocities & append outputs)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    do_append = APPEND or args.cont
    do_init_vel = False if args.cont else INIT_VELOCITIES

    calc = setup_calculator(args.model)
    atoms = setup_atoms(args.input, args.T, calc)

    if args.ensemble == "optimize":
        run_optimize(atoms, pressure=args.P)
    else:
        if do_init_vel:
            init_velocities(atoms, args.T)

        if args.ensemble == "nvt":
            run_nvt(
                atoms, args.T, args.dt, args.steps, args.interval,
                args.traj, args.out, append=do_append, tdamp=TDAMP,
            )
        elif args.ensemble == "npt_iso":
            run_npt_iso(
                atoms, args.T, args.P, args.dt, args.steps, args.interval,
                args.traj, args.out, append=do_append, tdamp=TDAMP, pdamp=PDAMP,
            )
        elif args.ensemble == "npt_aniso":
            run_npt_aniso(
                atoms, args.T, args.P, args.dt, args.steps, args.interval,
                args.traj, args.out, append=do_append, tdamp=TDAMP, pdamp=PDAMP,
                bulk_modulus_gpa=BULK_MODULUS_GPA,
            )
        elif args.ensemble == "npt_tri":
            run_npt_tri(
                atoms, args.T, args.P, args.dt, args.steps, args.interval,
                args.traj, args.out, append=do_append, tdamp=TDAMP, pdamp=PDAMP,
            )
        elif args.ensemble == "nve":
            run_nve(
                atoms, args.dt, args.steps, args.interval,
                args.traj, args.out, append=do_append,
            )

    atoms.wrap()
    write(args.final, atoms, format="vasp", vasp5=True, direct=True)
    print(f"\nDone! Trajectory: {args.traj} | Thermo log: {args.out} | Final structure: {args.final}")
