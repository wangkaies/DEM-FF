"""Validate explicit deepmd/npy E/F/V systems and write a native training input.

Does not convert labels, infer electronic temperature, split data, or run training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def inspect_system(path):
    path=Path(path).resolve(strict=True)
    names=(path/'type_map.raw').read_text().split()
    types=np.loadtxt(path/'type.raw',dtype=int).reshape(-1)
    if not names or len(set(names))!=len(names) or not len(types) or np.any(types<0) or np.any(types>=len(names)):
        raise ValueError(f'Invalid type map: {path}')
    sets=sorted(p for p in path.glob('set.*') if p.is_dir())
    if not sets:raise ValueError(f'No set.* directories: {path}')
    n=0;hashes=set()
    for subset in sets:
        a={k:np.load(subset/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ['coord','box','energy','force','virial','fparam']}
        nf=a['coord'].shape[0];na=len(types)
        for k,width in [('coord',na*3),('box',9),('energy',1),('force',na*3),('virial',9),('fparam',1)]:
            if a[k].ndim<1 or a[k].shape[0]!=nf or a[k].size!=nf*width or not np.isfinite(a[k]).all():
                raise ValueError(f'Invalid {k} shape/values: {subset}')
        if not nf or np.any(np.linalg.det(a['box'].reshape(-1,3,3))<=1e-10) or np.any(a['fparam']<0):
            raise ValueError(f'Empty system, invalid cell, or negative fparam: {subset}')
        n+=nf
        for xyz,cell in zip(a['coord'],a['box']):
            # Conservative duplicate diagnostic: same order and Cartesian origin.
            geometry=np.concatenate([xyz.reshape(-1),cell.reshape(-1)])
            quantized=np.rint(geometry*1e5).astype('<i8')
            key=hashlib.sha256(' '.join(names).encode()+types.astype('<i8').tobytes()+quantized.tobytes()).hexdigest()
            hashes.add(key)
    return names,n,hashes


def prepare(foundation,train,valid,output,steps=1000,lr=1e-5,seed=731):
    from deepmd_demff.manifest import load
    foundation=Path(foundation).resolve(strict=True);load(foundation)
    if steps<1 or not np.isfinite(lr) or lr<=0:raise ValueError('steps and lr must be positive')
    groups={'train':[Path(p).resolve(strict=True) for p in train],'valid':[Path(p).resolve(strict=True) for p in valid]}
    if not all(groups.values()) or set(groups['train']) & set(groups['valid']):raise ValueError('Distinct train and valid systems are required')
    common=None;seen={'train':set(),'valid':set()}
    for split,paths in groups.items():
        for path in paths:
            names,n,hashes=inspect_system(path)
            if common is None:common=names
            if names!=common:raise ValueError('All systems must use the same explicit type_map order')
            seen[split].update(hashes)
    if seen['train'] & seen['valid']:raise ValueError('Train/valid contain matching ordered Cartesian geometries')
    data={'model':{'type':'demff','foundation':str(foundation),'type_map':common,'precision':'float64'},
          'learning_rate':{'type':'exp','start_lr':lr,'stop_lr':lr,'decay_steps':100},
          'loss':{'type':'ener','start_pref_e':1.,'limit_pref_e':1.,'start_pref_f':1.,'limit_pref_f':1.,'start_pref_v':.01,'limit_pref_v':.01},
          'training':{'training_data':{'systems':[str(p) for p in groups['train']],'batch_size':1},
                      'validation_data':{'systems':[str(p) for p in groups['valid']],'batch_size':1,'numb_btch':1},
                      'numb_steps':steps,'seed':seed,'disp_freq':min(10,steps),'save_freq':steps,'save_ckpt':'model.ckpt','disp_file':'lcurve.out'}}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--foundation',required=True);p.add_argument('--train',nargs='+',required=True)
    p.add_argument('--valid',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--seed',type=int,default=731)
    a=p.parse_args();prepare(a.foundation,a.train,a.valid,a.output,a.steps,a.lr,a.seed)
    print('Wrote',a.output)


if __name__=='__main__':main()
