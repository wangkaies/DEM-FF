# Design, related work, and scope

## What this repository already provides

The existing `ase_DEMFF_simulation.py` provides CPU/CUDA selection, electronic
temperature, NVT, isotropic/anisotropic NPT, NVE, cell relaxation, trajectory
continuation, thermodynamic/stress logging, and final POSCAR output. The root
README also documents direct ASE inference and the model's LES contribution.
This integration does not claim any of these as new features and does not replace
the existing ASE example.

## Existing MACE integration

[DeePMD-GNN](https://github.com/deepmodeling/deepmd-gnn) already integrates MACE
with DeePMD training and MD workflows. Credit: J. Zeng et al.,
[DeePMD-GNN, JCIM (2025)](https://doi.org/10.1021/acs.jcim.4c02441).

At inspected revision `d60816fdd94c4269f039996ed9e1a7111238139e`, its
[MACE-OFF loader](https://github.com/deepmodeling/deepmd-gnn/blob/d60816fdd94c4269f039996ed9e1a7111238139e/deepmd_gnn/mace_off.py)
explicitly rejects joint-embedding checkpoints, and
[network construction](https://github.com/deepmodeling/deepmd-gnn/blob/d60816fdd94c4269f039996ed9e1a7111238139e/deepmd_gnn/mace_network.py)
constructs `ScaleShiftMACE`. This is source-level evidence for the distinction;
it is not a runtime comparison or a claim that every possible DeePMD-GNN extension
cannot support DEM-FF. No DeePMD-GNN source is copied into this adapter.

The contribution here is a deliberately narrow adapter for the existing complete
DEM-FF `MACELES` object and its `elec_temp` joint embedding. Short-range weights,
LES weights, original element coverage, and pretrained reference energies are
preserved. A future generalized MACELES implementation could belong in a shared
integration project; the isolated folder keeps that option open.

## Entry points and data flow

`deepmd.backend` registers `.demff` for `DeepPot` and `dp test`. The manifest
contains a relative model path, SHA256, units, and periodicity. `deepmd.pt`
registers `model.type = demff` for the native PyTorch trainer. Training is performed
by DeePMD itself; this package does not implement a separate optimizer loop.

`fparam[:, 0]` is electronic kBT in eV. MACE builds a complete periodic graph.
The adapter passes differentiable positions, cell, periodic shifts, and electronic
temperature to the original network. Energy derivatives produce forces and total
virial. During training, `create_graph=True` preserves the higher derivatives
required by force/virial losses to update the weights. The calculator's default
disabled parameter gradients are explicitly re-enabled for native training.

`get_rcut()` describes the local MACE graph cutoff, not a cutoff of LES physics.
It must not be used to infer that the full potential permits local MPI domains.
The `atomic_model.observed_type` metadata shim serves the pinned DeePMD trainer;
it does not imply atomic decomposition of LES energy.

## Virial convention and pinned implementation

In the inspected MACE 0.3.16 / LES 0.2.0 path, the internal strain handling
strains edge vectors while explicit positions/cell passed to LES remain
unstrained. Consequently the native stress result is not a complete reference
for this adapter's total energy strain derivative. The adapter applies symmetric
strain to positions, cell, and periodic shifts before evaluating the full model,
and computes `virial = -dE/dstrain`. The regression compares that derivative to
independent central finite differences, including the author's POSCAR.

Energy and forces are compared to the original MACE calculator at float64.
This note is version-specific and does not assert a defect in every MACE/LES
release. It does not alter the author's ASE script.

## Portability limits

Native `.pt` files save the trainable state and DeePMD optimizer metadata.
Reconstruction still reads the saved foundation path. `demff-deepmd export`
rebuilds the original architecture, loads the trained state, and saves a MACE
object plus a SHA256 manifest. A manifest is not a frozen standard DP model.
The underlying Python checkpoint must be trusted.

The private graph-building API, observed-type metadata shim, and pinned versions
are maintenance risks. Python full-system evaluation also favors correctness
checks over throughput. This proposal makes no C++ LAMMPS, multi-GPU scaling,
or long MD stability claim.
