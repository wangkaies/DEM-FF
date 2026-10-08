"""Differentiable full-system DEM-FF model for DeePMD's native PT Trainer.

No training loop or optimizer is implemented here. Neighbor topology is built
by pinned MACE; geometry and total-energy derivatives stay in the torch graph.
"""
import json
import logging
from types import SimpleNamespace
import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers
from deepmd.pt.model.model.model import BaseModel
from deepmd.pt.utils import env

from .manifest import load


@BaseModel.register("demff")
class DEMFFModel(BaseModel):
    def __init__(self, foundation, type_map, precision="float64", **kwargs):
        super().__init__()
        for key in ("atom_exclude_types", "pair_exclude_types", "spin", "modifier", "use_srtab"):
            if kwargs.get(key):
                raise NotImplementedError(f"DEM-FF does not support {key}")
        if precision != "float64":
            raise ValueError("Training prototype is validated with float64 only")
        _, path = load(foundation)
        from mace.calculators import MACECalculator
        self._calculator = MACECalculator(model_paths=str(path), device=str(env.DEVICE),
                                         default_dtype=precision,
                                         info_keys={"elec_temp": "elec_temp"})
        self.mace = self._calculator.models[0]
        if self.mace.__class__.__name__ != "MACELES" or "elec_temp" not in self.mace.embedding_specs:
            raise ValueError("Expected a complete DEM-FF MACELES checkpoint")
        self.type_map = list(type_map)
        # Trainer reads this metadata path even for external full-system models.
        # This is not an atomic-energy model or a decomposition of LES energy.
        self.atomic_model = SimpleNamespace(observed_type=kwargs.get("info",{}).get("observed_type"))
        if len(set(type_map)) != len(type_map):
            raise ValueError("type_map contains duplicates")
        available = set(self.mace.atomic_numbers.detach().cpu().tolist())
        numbers = [atomic_numbers[s] for s in type_map]
        if not set(numbers) <= available:
            raise ValueError("Training elements are absent from the pretrained model")
        self.numbers = numbers
        self.rcut = float(self.mace.r_max.detach().cpu())
        self.foundation = str(foundation)
        self.precision = precision
        # MACECalculator disables parameter gradients for inference. Restore
        # them explicitly: both short-range and LES weights must be trainable.
        self.mace.requires_grad_(True)

    def forward(self, coord, atype, box=None, fparam=None, aparam=None,
                do_atomic_virial=False, charge_spin=None):
        if box is None or fparam is None:
            raise ValueError("DEM-FF requires a periodic box and electronic kBT (eV) in fparam")
        if do_atomic_virial or aparam is not None or charge_spin is not None:
            raise NotImplementedError("Atomic virial, atomic parameters and charge/spin inputs are unsupported")
        # DeePMD's loader may keep box/fparam on CPU while coordinates are CUDA.
        device = self.mace.atomic_numbers.device
        coord = coord.to(device=device,dtype=torch.float64)
        box = box.to(device=device,dtype=torch.float64)
        fparam = fparam.to(device=device,dtype=torch.float64)
        atype = atype.to(device=device)
        if torch.any(atype < 0) or torch.any(atype >= len(self.type_map)):
            raise ValueError("Invalid atom types; padding/virtual atoms are unsupported")
        nf, na = atype.shape
        coord, box = coord.reshape(nf,na,3), box.reshape(nf,3,3)
        fparam = fparam.reshape(nf,1)
        if not all(torch.isfinite(v).all() for v in (coord,box,fparam)) or torch.any(fparam < 0):
            raise ValueError("Non-finite inputs or negative electronic kBT")
        if torch.any(torch.linalg.det(box) <= 1e-10):
            raise ValueError("Full right-handed periodic cells are required")
        energies, forces, virials = [], [], []
        with torch.enable_grad():
            for frame in range(nf):
                xyz = coord[frame].detach().clone().requires_grad_(True)
                cell = box[frame]
                atoms = Atoms(numbers=[self.numbers[int(i)] for i in atype[frame].detach().cpu()],
                              positions=xyz.detach().cpu().numpy(),
                              cell=cell.detach().cpu().numpy(), pbc=True)
                atoms.info["elec_temp"] = float(fparam[frame,0].detach().cpu())
                graph = self._calculator._atoms_to_batch(atoms).to_dict()
                # Tensor device/dtype follows the trainable module, not cached
                # calculator settings; this also supports state reload on CPU.
                device = self.mace.atomic_numbers.device
                graph = {k:v.to(device=device, dtype=torch.float64 if v.is_floating_point() else v.dtype)
                         if torch.is_tensor(v) else v for k,v in graph.items()}
                strain = torch.zeros((1,3,3), dtype=xyz.dtype, device=xyz.device, requires_grad=True)
                transform = torch.eye(3, dtype=xyz.dtype, device=xyz.device)[None] + (strain+strain.transpose(-1,-2))*0.5
                strained_cell = cell[None] @ transform
                graph["positions"] = xyz @ transform[0]
                graph["cell"] = strained_cell.reshape(-1,3)
                graph["shifts"] = graph["unit_shifts"] @ strained_cell[0]
                graph["elec_temp"] = fparam[frame:frame+1].reshape_as(graph["elec_temp"])
                result = self.mace(graph, training=False, compute_force=False, compute_stress=False)
                energy = result["energy"].sum()
                # create_graph is essential for force/virial losses to update
                # neural-network weights through these first derivatives.
                grad_xyz, grad_strain = torch.autograd.grad(
                    energy, (xyz,strain), create_graph=self.training, retain_graph=self.training)
                energies.append(energy.reshape(1))
                forces.append(-grad_xyz)
                virials.append(-grad_strain.reshape(9))
        return {"energy":torch.stack(energies), "force":torch.stack(forces),
                "virial":torch.stack(virials)}

    def get_type_map(self): return list(self.type_map)
    def get_rcut(self): return self.rcut
    def get_dim_fparam(self): return 1
    def get_dim_aparam(self): return 0
    def get_sel_type(self): return []
    def is_aparam_nall(self): return False
    def model_output_type(self): return ["energy"]
    def get_nnei(self): return 0
    def get_nsel(self): return 0
    def mixed_types(self): return True
    def has_default_fparam(self): return False
    def get_observed_type_list(self): return list(self.atomic_model.observed_type or [])
    def has_message_passing(self): return False
    def need_sorted_nlist_for_lower(self): return False

    @classmethod
    def update_sel(cls, train_data, type_map, local_jdata):
        return dict(local_jdata), None  # MACE constructs complete periodic graphs.

    def compute_or_load_stat(self, sampled_func, stat_file_path=None, preset_observed_type=None):
        if preset_observed_type is not None:
            self.atomic_model.observed_type = list(preset_observed_type)
        else:
            observed = set()
            for system in sampled_func():
                atype = system["atype"]
                if torch.is_tensor(atype): atype=atype.detach().cpu().numpy()
                observed.update(np.unique(atype).tolist())
            self.atomic_model.observed_type = [s for i,s in enumerate(self.type_map) if i in observed]
        logging.info("DEM-FF: preserving all pretrained normalization and atomic reference energies")

    def get_out_bias(self):
        indices = [self.mace.atomic_numbers.tolist().index(z) for z in self.numbers]
        return self.mace.atomic_energies_fn.atomic_energies[...,indices].detach().clone()

    def change_out_bias(self, sample_func, bias_adjust_mode="change-by-statistic"):
        logging.info("DEM-FF: retaining pretrained atomic reference energies during fine-tune; no automatic offset fit")

    def change_type_map(self, type_map, **kwargs):
        if list(type_map) != self.type_map:
            raise NotImplementedError("Checkpoint fine-tuning currently requires the same type_map")

    def serialize(self):
        raise NotImplementedError("Use native DeePMD .pt checkpoints and demff-deepmd export; portable DP serialization is not supported")
