"""Generate small synthetic fixtures using the user's local DEM-FF checkpoint.

No private research data or pretrained weights are bundled. These are software
tests, not DFT labels, equilibrium structures, or a scientific training recipe.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from deepmd.infer import DeepPot


def generate(manifest,output,device='cuda'):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    dp=DeepPot(str(Path(manifest).resolve()),device=device,default_dtype='float64')
    symbols=['O','H','H','O','H','H'];types=[dp.get_type_map().index(s) for s in symbols]
    base=np.array([[1,1,1],[1.96,1,1],[.76,1.93,1],[3.6,3.5,3.4],[4.54,3.6,3.4],[3.35,4.41,3.6]])
    cell=np.array([[7.,0,0],[.5,7.2,0],[.3,.4,7.4]])
    rng=np.random.default_rng(20260912)
    for split,n in [('train',6),('valid',2)]:
        d=root/split/'set.000';d.mkdir(parents=True)
        coord=base[None]+rng.normal(0,.02,(n,6,3));box=np.tile(cell,(n,1,1))
        kbt=np.linspace(.0861733,.1723466,n).reshape(-1,1)
        energy,force,virial=dp.eval(coord,box,types,fparam=kbt)
        # A constant synthetic energy shift preserves force/virial derivatives
        # and makes the training smoke exercise a nonzero optimizer update.
        energy=energy+.05
        for key,a in [('coord',coord.reshape(n,-1)),('box',box.reshape(n,9)),('energy',energy.reshape(n)),('force',force.reshape(n,-1)),('virial',virial.reshape(n,9)),('fparam',kbt)]:
            np.save(d/(key+'.npy'),a)
        (d.parent/'type_map.raw').write_text('P\nO\nH\n')
        np.savetxt(d.parent/'type.raw',[1,2,2,1,2,2],fmt='%d')
        (d.parent/'LABEL_PROVENANCE.txt').write_text('SYNTHETIC: DEM-FF E/F/V; energy has a constant +0.05 eV shift. Not DFT; no physical accuracy claim.\n')
    (root/'provenance.json').write_text(json.dumps({'purpose':'software smoke test only','atoms':6,'elements_exercised':['O','H'],'seed':20260912,'energy_shift_eV':.05},indent=2))
    return root


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest');p.add_argument('output');p.add_argument('--device',default='cuda')
    a=p.parse_args();print(generate(a.manifest,a.output,a.device))
