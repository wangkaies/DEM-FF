"""Export a native DeePMD training checkpoint back to its MACE-LES architecture."""
from pathlib import Path
import torch
from .manifest import register


def export_checkpoint(checkpoint, output):
    from deepmd.pt.model.model import get_model
    from deepmd.pt.train.wrapper import ModelWrapper
    output=Path(output).resolve()
    if output.suffix!='.demff': raise ValueError('Output must end in .demff')
    weight=output.with_suffix('.model')
    if output.exists() or weight.exists():raise FileExistsError('Export refuses to overwrite existing model files')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    params=state['model']['_extra_state']['model_params']
    if params.get('type')!='demff':raise ValueError('Not a DEM-FF training checkpoint')
    wrapper=ModelWrapper(get_model(params),model_params=params)
    wrapper.load_state_dict(state['model'])
    mace=wrapper.model['Default'].mace.cpu().eval()
    mace.requires_grad_(False)
    torch.save(mace,weight)
    return register(weight,output)
