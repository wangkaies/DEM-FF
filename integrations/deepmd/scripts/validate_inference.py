"""Validate the real checkpoint through public DeePMD APIs (not a DFT accuracy test)."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from ase import Atoms
from ase.io import read
from deepmd.infer import DeepPot
from mace.calculators import MACECalculator

from deepmd_demff.manifest import digest, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("poscar")
    parser.add_argument("output")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Allocated GPU is not visible to torch")
    _, weight = load(args.manifest)
    report = {"purpose": "interface parity and derivative verification; NOT DFT accuracy",
              "device": args.device, "model_sha256": digest(weight), "checks": {},
              "versions": {x: importlib.metadata.version(x) for x in
                           ("torch", "deepmd-kit", "mace-torch", "les", "ase", "deepmd-demff")}}
    if args.device == "cuda":
        report["gpu"] = torch.cuda.get_device_name(0)
    dp = DeepPot(args.manifest, device=args.device, default_dtype="float64")
    reference = MACECalculator(model_paths=str(weight), device=args.device,
                              default_dtype="float64", info_keys={"elec_temp": "elec_temp"})
    type_map = dp.get_type_map()
    report["type_map"] = type_map
    assert dp.get_dim_fparam() == 1 and dp.get_dim_aparam() == 0

    def predict(atoms, kbt):
        atype = [type_map.index(x) for x in atoms.get_chemical_symbols()]
        return dp.eval(atoms.positions[None], atoms.cell.array[None], atype, fparam=[[kbt]])

    def native(atoms, kbt):
        atoms = atoms.copy()
        atoms.info["elec_temp"] = float(kbt)
        reference.reset()
        atoms.calc = reference
        return (np.array([[atoms.get_potential_energy()]]), atoms.get_forces()[None],
                (-atoms.get_volume() * atoms.get_stress(voigt=False)).reshape(1, 9))

    def compare(label, actual, expected, atol=1e-8, rtol=1e-9, native_stress=False):
        errors = {}
        for key, got, want in zip(("energy_eV", "force_eV_A", "virial_eV"), actual, expected):
            assert np.isfinite(got).all(), (label, key, "not finite")
            if not (native_stress and key == "virial_eV"):
                np.testing.assert_allclose(got, want, atol=atol, rtol=rtol, err_msg=label + ":" + key)
            errors[key] = float(np.max(np.abs(got - want)))
        if native_stress:
            errors["native_virial_parity_required"] = False
            errors["reason"] = "MACE 0.3.16 native stress omits explicit LES strain; corrected virial checked by finite differences"
        report["checks"][label] = errors
        print(label, errors, flush=True)

    # Complete author-provided 350-atom system, with all its element mappings.
    # Read coordinates only: original POSCAR may contain a nonstandard velocity tail.
    lines = Path(args.poscar).read_text().splitlines()
    n = sum(map(int, lines[6].split()))
    import io
    original = read(io.StringIO("\n".join(lines[:8+n]) + "\n"), format="vasp")
    report["author_system_natoms"] = len(original)
    compare("author_350_atoms", predict(original, 4000 * 8.61733e-5),
            native(original, 4000 * 8.61733e-5), native_stress=True)
    author_virial = predict(original, 4000 * 8.61733e-5)[2].reshape(3,3)
    author_fd_errors = []
    for i,j in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)):
        direction = np.zeros((3,3))
        direction[i,j] = direction[j,i] = 1 if i==j else 0.5
        plus, minus = original.copy(), original.copy()
        plus.set_cell(original.cell.array @ (np.eye(3)+1e-5*direction), scale_atoms=True)
        minus.set_cell(original.cell.array @ (np.eye(3)-1e-5*direction), scale_atoms=True)
        fd = -float((native(plus,4000*8.61733e-5)[0]-native(minus,4000*8.61733e-5)[0]).item())/2e-5
        author_fd_errors.append(abs(fd-author_virial[i,j]))
    assert max(author_fd_errors) < 2e-3, ("author strain finite differences",author_fd_errors)
    report["checks"]["author_350_virial_fd_abs_errors_eV"] = author_fd_errors
    print("author_350_virial_fd_abs_errors_eV",author_fd_errors,flush=True)

    # Compact periodic interface fixture. It is not a sampled physical equilibrium state.
    atoms = Atoms("OHHOHH", positions=[[1,1,1],[1.96,1,1],[0.76,1.93,1],
                  [3.6,3.5,3.4],[4.54,3.6,3.4],[3.35,4.41,3.6]],
                  cell=[[7,0,0],[0.5,7.2,0],[0.3,0.4,7.4]], pbc=True)
    kbt = 2000 * 8.61733e-5
    baseline = predict(atoms, kbt)
    compare("triclinic", baseline, native(atoms, kbt), native_stress=True)
    rotated = atoms.copy()
    angle = 0.37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    rotated.positions = atoms.positions @ rotation.T
    rotated.set_cell(atoms.cell.array @ rotation.T)
    rr = predict(rotated, kbt)
    rv = rotation @ baseline[2].reshape(3, 3) @ rotation.T
    compare("rotation_covariance", rr, (baseline[0], baseline[1] @ rotation.T, rv.reshape(1,9)), atol=2e-7)
    permutation = np.array([5, 0, 2, 3, 1, 4])
    pr = predict(atoms[permutation], kbt)
    compare("atom_permutation", pr, (baseline[0], baseline[1][:, permutation], baseline[2]), atol=2e-7)
    shifted = atoms.copy()
    shifted.positions[1] += atoms.cell.array[0]
    compare("periodic_image", predict(shifted, kbt), baseline, atol=2e-7)

    # Same geometry at different electron temperatures must not hit a stale ASE cache.
    temps = np.array([kbt, 6000 * 8.61733e-5, kbt])
    types = [type_map.index(x) for x in atoms.get_chemical_symbols()]
    batched = dp.eval(np.tile(atoms.positions[None], (3,1,1)),
                      np.tile(atoms.cell.array[None], (3,1,1)), types, fparam=temps[:,None])
    natives = [native(atoms, t) for t in temps]
    compare("temperature_batch", batched, tuple(np.concatenate([x[j] for x in natives]) for j in range(3)), native_stress=True)
    delta_temp = float(abs(batched[0][1,0] - batched[0][0,0]))
    assert delta_temp > 1e-10, "electron-temperature response vanished"
    report["checks"]["temperature_energy_change_eV"] = delta_temp

    # LES must contribute to total energy. MACE node energies omit this term.
    model = dp.get_model()
    from mace import data
    from mace.tools import torch_geometric
    raw_atoms = atoms.copy()
    raw_atoms.info["elec_temp"] = kbt
    from mace.tools import torch_tools
    with torch_tools.default_dtype("float64"):
        config = data.config_from_atoms(raw_atoms, key_specification=data.KeySpecification(info_keys={"elec_temp":"elec_temp"}))
        graph = data.AtomicData.from_config(config, z_table=dp.deep_eval.calc.z_table, cutoff=dp.get_rcut(), heads=model.heads)
    batch = torch_geometric.batch.Batch.from_data_list([graph]).to(args.device)
    raw_inputs = {key: value.to(dtype=next(model.parameters()).dtype)
                  if isinstance(value, torch.Tensor) and value.is_floating_point() else value
                  for key, value in batch.to_dict().items()}
    raw = model(raw_inputs, training=False, compute_force=True, compute_stress=True)
    les = float(raw["les_energy"].detach().cpu().sum())
    total = float(raw["energy"].detach().cpu().sum())
    short = float(raw["node_energy"].detach().cpu().sum())
    np.testing.assert_allclose(total-short, les, atol=1e-8)
    np.testing.assert_allclose(baseline[0][0,0], total, atol=1e-8)
    assert abs(les) > 1e-10, "LES check needs a nonzero long-range contribution"
    report["checks"]["les_energy_eV"] = les

    # Central finite differences independently check force and virial conventions.
    h = 1e-4
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions[1, 0] += h
    minus.positions[1, 0] -= h
    fd_force = -float((predict(plus,kbt)[0] - predict(minus,kbt)[0]).item()) / (2*h)
    force_error = abs(fd_force - baseline[1][0,1,0])
    assert force_error < 2e-4, ("force finite difference", force_error)
    report["checks"]["force_fd_abs_error_eV_A"] = float(force_error)
    strain_errors = []
    h = 1e-5
    for i, j in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)):
        direction = np.zeros((3,3))
        direction[i,j] = 1 if i == j else 0.5
        direction[j,i] = direction[i,j]
        plus, minus = atoms.copy(), atoms.copy()
        plus.set_cell(atoms.cell.array @ (np.eye(3)+h*direction), scale_atoms=True)
        minus.set_cell(atoms.cell.array @ (np.eye(3)-h*direction), scale_atoms=True)
        fd_v = -float((predict(plus,kbt)[0] - predict(minus,kbt)[0]).item()) / (2*h)
        strain_errors.append(abs(fd_v-baseline[2].reshape(3,3)[i,j]))
    assert max(strain_errors) < 2e-3, ("strain finite differences",strain_errors)
    report["checks"]["virial_fd_abs_errors_eV"] = strain_errors

    rejected = []
    for name, kw in (("missing_temperature", {}), ("atomic_outputs", {"fparam":[[kbt]],"atomic":True}),
                     ("negative_temperature", {"fparam":[[-1.0]]})):
        try:
            dp.eval(atoms.positions[None], atoms.cell.array[None], types, **kw)
        except (ValueError, NotImplementedError):
            rejected.append(name)
        else:
            raise AssertionError("Unsupported input accepted: " + name)
    report["checks"]["rejected_inputs"] = rejected

    # Standard dp test: independent native ASE E/F, finite-difference virial.
    dataset = output / "ase_reference_dataset"
    subset = dataset / "set.000"
    subset.mkdir(parents=True, exist_ok=True)
    np.savetxt(dataset/"type.raw", types, fmt="%d")
    (dataset/"type_map.raw").write_text("\n".join(type_map)+"\n")
    np.save(subset/"coord.npy", np.tile(atoms.positions.reshape(1,-1), (3,1)))
    np.save(subset/"box.npy", np.tile(atoms.cell.array.reshape(1,9), (3,1)))
    np.save(subset/"fparam.npy", temps[:,None])
    for field,j in (("energy",0),("force",1)):
        np.save(subset/(field+".npy"), np.concatenate([x[j] for x in natives]).reshape(3,-1))
    native_fd_virials = []
    for temp in temps:
        tensor = np.zeros((3,3))
        for i,j in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)):
            direction = np.zeros((3,3))
            direction[i,j] = direction[j,i] = 1 if i==j else 0.5
            plus, minus = atoms.copy(), atoms.copy()
            plus.set_cell(atoms.cell.array @ (np.eye(3)+h*direction), scale_atoms=True)
            minus.set_cell(atoms.cell.array @ (np.eye(3)-h*direction), scale_atoms=True)
            tensor[i,j] = tensor[j,i] = -float((native(plus,temp)[0]-native(minus,temp)[0]).item())/(2*h)
        native_fd_virials.append(tensor.reshape(9))
    np.save(subset/"virial.npy", np.asarray(native_fd_virials))
    (dataset/"LABEL_PROVENANCE.txt").write_text("Native DEM-FF ASE energies/forces; virials from central finite differences of native ASE energy. NOT DFT labels.\n")
    cli = subprocess.run([sys.executable,"-m","deepmd","test","-m",str(Path(args.manifest).resolve()),
                          "-s",str(dataset),"-n","3","-d",str(output/"dp_test")],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output/"dp-test.log").write_text(cli.stdout)
    assert cli.returncode == 0, cli.stdout
    report["checks"]["dp_test_exit_code"] = cli.returncode
    cli_errors = {}
    for field in ("e", "f", "v"):
        values = np.loadtxt(output / ("dp_test." + field + ".out"), ndmin=2)
        left, right = np.split(values, 2, axis=1)
        np.testing.assert_allclose(left, right, atol=2e-3 if field=="v" else 2e-7, rtol=1e-9,
                                   err_msg="dp test native ASE parity: " + field)
        cli_errors[field] = float(np.max(np.abs(left - right)))
    report["checks"]["dp_test_max_abs_errors"] = cli_errors
    report["elapsed_seconds"] = time.time()-start
    report["status"] = "PASS"
    (output/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report,indent=2), flush=True)


if __name__ == "__main__":
    main()
