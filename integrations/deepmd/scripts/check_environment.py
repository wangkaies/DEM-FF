"""Print runtime versions without changing the environment."""
from importlib.metadata import version
import json
import platform
import torch
import deepmd_demff
import deepmd_demff.pt

report={'python':platform.python_version(),'plugin_source':deepmd_demff.__file__,
        'versions':{p:version(p) for p in ['deepmd-demff','deepmd-kit','mace-torch','e3nn','ase','les','torch']},
        'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda}
if report['cuda_available']:report['gpu']=torch.cuda.get_device_name(0)
print(json.dumps(report,indent=2))
