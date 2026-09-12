"""Full-system periodic inference. This is not a C++/LAMMPS runtime backend."""

import numpy as np
from deepmd.infer.deep_eval import DeepEvalBackend

from .manifest import load


class DEMFFEval(DeepEvalBackend):
    def __init__(self, model_file, output_def, *args, auto_batch_size=True,
                 neighbor_list=None, device="cpu", default_dtype="float64", head=None, **kwargs):
        if args or kwargs:
            raise TypeError(f"Unsupported DEM-FF constructor arguments: {args}, {list(kwargs)}")
        if neighbor_list is not None:
            raise NotImplementedError("External neighbor lists are unsupported: LES needs the complete cell")
        if head is not None:
            raise NotImplementedError("This checkpoint bridge uses its default model head")
        if default_dtype not in ("float32", "float64"):
            raise ValueError("default_dtype must be float32 or float64")
        self.output_def = output_def
        self.metadata, model_path = load(model_file)
        from ase.data import chemical_symbols
        from mace.calculators import MACECalculator
        self.calc = MACECalculator(model_paths=str(model_path), device=device,
                                   default_dtype=default_dtype,
                                   info_keys={"elec_temp": "elec_temp"})
        model = self.calc.models[0]
        if model.__class__.__name__ != "MACELES":
            raise TypeError(f"Expected original MACELES, got {type(model)}")
        specs = getattr(model, "embedding_specs", None)
        if not specs or "elec_temp" not in specs:
            raise ValueError("Checkpoint does not declare the DEM-FF elec_temp embedding")
        self.atomic_numbers = np.asarray(model.atomic_numbers.detach().cpu(), dtype=int)
        self.type_map = [chemical_symbols[z] for z in self.atomic_numbers]
        self.rcut = float(model.r_max.detach().cpu())

    def _evaluate_atoms(self, atoms):
        import torch
        # MACE 0.3.16's internal stress path does not strain the explicit LES
        # positions/cell. Differentiate total energy through an external strain
        # instead; retain the original model, parameters and periodic graph.
        with torch.enable_grad():
            batch = self.calc._atoms_to_batch(atoms).to_dict()
            model = self.calc.models[0]
            dtype = next(model.parameters()).dtype
            batch = {k: v.to(dtype=dtype) if torch.is_tensor(v) and v.is_floating_point() else v
                     for k, v in batch.items()}
            positions = batch["positions"].detach().requires_grad_(True)
            strain = torch.zeros((1, 3, 3), dtype=dtype, device=positions.device,
                                 requires_grad=True)
            symmetric = (strain + strain.transpose(-1, -2)) * 0.5
            transform = torch.eye(3, dtype=dtype, device=positions.device)[None] + symmetric
            cells = batch["cell"].reshape(1, 3, 3) @ transform
            batch["positions"] = torch.einsum("ni,nij->nj", positions, transform[batch["batch"]])
            batch["cell"] = cells.reshape(-1, 3)
            senders = batch["edge_index"][0]
            batch["shifts"] = torch.einsum("ni,nij->nj", batch["unit_shifts"], cells[batch["batch"][senders]])
            raw = model(batch, training=False, compute_force=False, compute_stress=False)
            energy = raw["energy"].sum()
            grad_pos, grad_strain = torch.autograd.grad(energy, (positions, strain))
            return (float(energy.detach().cpu()), -grad_pos.detach().cpu().numpy(),
                    -grad_strain[0].detach().cpu().numpy())

    def eval(self, coords, cells, atom_types, atomic=False, fparam=None, aparam=None, **kwargs):
        if atomic:
            raise NotImplementedError("Atomic energy/virial decomposition is not defined for this LES bridge")
        if aparam is not None and np.asarray(aparam).size:
            raise ValueError("DEM-FF has no atomic parameters")
        if any(value is not None for value in kwargs.values()):
            raise NotImplementedError(f"Unsupported DEM-FF evaluation options: {list(kwargs)}")
        if cells is None:
            raise NotImplementedError("This bridge requires a full 3D periodic cell")
        if fparam is None:
            raise ValueError("fparam is required: electronic kBT in eV, e.g. [[4000 * 8.61733e-5]]")
        coords = np.asarray(coords, dtype=float)
        types = np.asarray(atom_types)
        if not np.issubdtype(types.dtype, np.integer):
            raise ValueError("Atom type indices must be integers")
        if types.size == 0:
            raise ValueError("Empty atom arrays are unsupported")
        natoms = types.shape[-1]
        coords = coords.reshape(-1, natoms, 3)
        nframes = len(coords)
        types = types.reshape(-1, natoms)
        if len(types) == 1:
            types = np.repeat(types, nframes, axis=0)
        if len(types) != nframes or np.any(types < 0) or np.any(types >= self.get_ntypes()):
            raise ValueError("Atom types have invalid shape or indices")
        cells = np.asarray(cells, dtype=float).reshape(-1, 3, 3)
        params = np.asarray(fparam, dtype=float).reshape(-1)
        if params.size == 1:
            params = np.repeat(params, nframes)
        if len(cells) != nframes or params.size != nframes:
            raise ValueError("One periodic cell and one electronic temperature are required per frame")
        if not all(np.isfinite(x).all() for x in (coords, cells, params)) or np.any(params < 0):
            raise ValueError("Inputs must be finite; electronic kBT must be nonnegative")
        if np.any(np.linalg.det(cells) <= 1e-10):
            raise ValueError("Cells must have positive nonzero volume")
        from ase import Atoms
        energies, forces, virials = [], [], []
        for xyz, cell, atype, kbt in zip(coords, cells, types, params):
            atoms = Atoms(numbers=self.atomic_numbers[atype], positions=xyz, cell=cell, pbc=True)
            atoms.info["elec_temp"] = float(kbt)
            energy, force, virial = self._evaluate_atoms(atoms)
            if not np.isfinite(energy) or not np.isfinite(force).all() or not np.isfinite(virial).all():
                raise FloatingPointError("MACE-LES returned non-finite energy, force, or virial")
            energies.append(energy)
            forces.append(force)
            virials.append(virial)
        return {"energy_redu": np.array(energies).reshape(nframes, 1),
                "energy_derv_r": np.array(forces).reshape(nframes, natoms, 3),
                "energy_derv_c_redu": np.array(virials).reshape(nframes, 9)}

    def get_rcut(self):
        """MACE local cutoff only. LES itself is not finite-range."""
        return self.rcut

    def get_ntypes(self):
        return len(self.type_map)

    def get_type_map(self):
        return list(self.type_map)

    def get_dim_fparam(self):
        return 1

    def get_dim_aparam(self):
        return 0

    def get_sel_type(self):
        return []

    def get_ntypes_spin(self):
        return 0

    @property
    def model_type(self):
        from deepmd.infer.deep_pot import DeepPot
        return DeepPot

    def get_model(self):
        return self.calc.models[0]

    def get_model_def_script(self):
        return {"model": {"type": "demff", "type_map": self.get_type_map(), "dim_fparam": 1},
                "inference_only": True, "fparam_units": "eV", "long_range": "LES"}
