"""Engineering validation of actual dp --pt train on public synthetic water fixtures."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from make_smoke_dataset import generate
from prepare_input import prepare

import numpy as np
import torch
from deepmd.pt.model.model import get_model
from deepmd.pt.train.wrapper import ModelWrapper
from deepmd.pt.loss.ener import EnergyStdLoss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("foundation")
    parser.add_argument("output")
    args=parser.parse_args()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    report={"status":"IN_PROGRESS","purpose":"software training integration, NOT scientific acceptance",
            "conditioning":"explicit synthetic kBT in eV; labels are NOT DFT",
            "checks":{}}
    def save(): (root/'training-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    torch.set_num_threads(4)
    params={"type":"demff","foundation":str(Path(args.foundation).resolve()),
            "type_map":["P","O","H"],"precision":"float64"}
    model=get_model(params)
    assert model.__class__.__name__=='DEMFFModel'
    device=next(model.parameters()).device
    assert device.type=='cuda'
    report['gpu']=torch.cuda.get_device_name(0)
    report['checks']['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    xyz=torch.tensor([[[1,1,1],[1.96,1,1],[0.76,1.93,1],[3.6,3.5,3.4],[4.54,3.6,3.4],[3.35,4.41,3.6]]],dtype=torch.float64,device=device)
    inputs={'coord':xyz,'atype':torch.tensor([[1,2,2,1,2,2]],device=device),
            'box':torch.tensor([[[7,0,0],[.5,7.2,0],[.3,.4,7.4]]],dtype=torch.float64,device=device),
            'fparam':torch.tensor([[0.1723466]],dtype=torch.float64,device=device)}
    model.eval();baseline=model(**inputs)
    # Independent scalar derivative of each native DeePMD loss w.r.t. a LES
    # neural-network parameter; this detects detached force/virial training.
    labels={k:v.detach()+({'energy':.1,'force':.03,'virial':.05}[k]) for k,v in baseline.items()}
    # A uniform force offset contracts with total force, which is identically
    # zero by translation invariance. Use a nonuniform residual instead.
    labels['force']=baseline['force'].detach()+.03*torch.sin(torch.arange(18,device=device,dtype=torch.float64)).reshape(1,6,3)
    labels.update({f'find_{k}':torch.tensor(1.,device=device) for k in baseline})
    model.train()
    for key,code in [('energy','e'),('force','f'),('virial','v')]:
        kw={f'{stage}_pref_{c}':float(c==code) for stage in ['start','limit'] for c in ['e','f','v']}
        loss_fn=EnergyStdLoss(starter_learning_rate=1e-4,**kw)
        model.zero_grad(set_to_none=True)
        _,loss,_=loss_fn(inputs,model,labels,natoms=6,learning_rate=1e-4)
        loss.backward()
        groups={}
        for prefix in ['mace.interactions','mace.les_readouts','mace.joint_embedding']:
            grads=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads),(key,prefix)
            groups[prefix]=float(sum(g.square().sum() for g in grads).sqrt())
            assert groups[prefix]>1e-12,(key,prefix,'zero gradient')
        name,param=max(((n,p) for n,p in model.named_parameters() if n.startswith('mace.les_readouts') and p.grad is not None),key=lambda x:float(x[1].grad.abs().max()))
        index=int(param.grad.abs().argmax());analytic=float(param.grad.flatten()[index]);old=float(param.detach().flatten()[index]);h=1e-5
        def shifted(sign):
            with torch.no_grad():param.flatten()[index]=old+sign*h
            _,value,_=loss_fn(inputs,model,labels,natoms=6,learning_rate=1e-4)
            return float(value.detach())
        try: fd=(shifted(1)-shifted(-1))/(2*h)
        finally:
            with torch.no_grad():param.flatten()[index]=old
        np.testing.assert_allclose(analytic,fd,rtol=2e-3,atol=1e-6)
        report['checks'][key+'_gradient']={'group_norms':groups,'parameter':name,'index':index,'analytic':analytic,'finite_difference':fd}
        print('GRADIENT_PASS',key,groups,flush=True);save()
    model.zero_grad(set_to_none=True)
    dataset=generate(args.foundation,root/'dataset',device='cuda')
    report['data_source']=json.loads((dataset/'provenance.json').read_text())
    config=prepare(args.foundation,[dataset/'train'],[dataset/'valid'],root/'prepared-input.json',steps=12,lr=1e-6)
    config['training'].update(disp_freq=1,save_freq=6)
    def run_cli(folder,config,extra=()):
        folder.mkdir(exist_ok=True)
        (folder/'input.json').write_text(json.dumps(config,indent=2))
        command=[sys.executable,'-m','deepmd','--pt','train','input.json',*extra]
        with (folder/'train.log').open('w') as fp:
            result=subprocess.run(command,cwd=folder,stdout=fp,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError((folder/'train.log').read_text()[-10000:])
        print('NATIVE_DP_TRAIN_PASS',str(folder),flush=True)
    run_cli(root/'initial',config)
    def checkpoint(path):
        state=torch.load(path,map_location=device,weights_only=True)
        wrapped=ModelWrapper(get_model(params),model_params=params)
        wrapped.load_state_dict(state['model']);wrapped.eval()
        return state,wrapped.model['Default']
    first_path=root/'initial/model.ckpt.pt'
    first,trained=checkpoint(first_path)
    original_state=model.state_dict();trained_state=trained.state_dict()
    deltas={}
    for prefix in ['mace.interactions','mace.les_readouts','mace.joint_embedding']:
        delta=max(float((trained_state[n]-v).abs().max()) for n,v in original_state.items() if n.startswith(prefix) and v.is_floating_point() and v.numel())
        assert delta>0,prefix;deltas[prefix]=delta
    report['checks']['weight_updates']=deltas
    assert first['optimizer']['state'],'optimizer state missing'
    # Compare the same six frames before/after, rather than comparing shuffled
    # single-batch lcurve rows. This remains an engineering check at synthetic labels.
    train_dir=dataset/'train/set.000'
    batch_inputs={'coord':torch.as_tensor(np.load(train_dir/'coord.npy'),device=device),
                  'box':torch.as_tensor(np.load(train_dir/'box.npy'),device=device),
                  'fparam':torch.as_tensor(np.load(train_dir/'fparam.npy'),device=device),
                  'atype':torch.as_tensor(np.tile(np.loadtxt(dataset/'train/type.raw',dtype=int),(6,1)),device=device)}
    batch_labels={k:torch.as_tensor(np.load(train_dir/(k+'.npy')).reshape(6,-1),device=device) for k in ['energy','force','virial']}
    batch_labels['force']=batch_labels['force'].reshape(6,-1,3)
    batch_labels.update({f'find_{k}':torch.tensor(1.,device=device) for k in ['energy','force','virial']})
    real_loss=EnergyStdLoss(starter_learning_rate=1e-6,start_pref_e=1.,limit_pref_e=1.,start_pref_f=1.,limit_pref_f=1.,start_pref_v=.01,limit_pref_v=.01)
    model.eval();trained.eval()
    before=float(real_loss(batch_inputs,model,batch_labels,natoms=6,learning_rate=1e-6)[1].detach())
    after=float(real_loss(batch_inputs,trained,batch_labels,natoms=6,learning_rate=1e-6)[1].detach())
    assert np.isfinite([before,after]).all() and after<before,(before,after)
    report['checks']['same_six_frame_loss']={'before':before,'after':after}
    print('SAME_FRAME_LOSS_DECREASE',before,after,flush=True);save()
    # Restart must retain optimizer state and advance the original step counter.
    resumed=copy.deepcopy(config);resumed['training']['numb_steps']=16
    run_cli(root/'restart',resumed,['--restart',str(first_path)])
    restarted,restart_model=checkpoint(root/'restart/model.ckpt.pt')
    report['checks']['checkpoint_extra_initial']=first['model']['_extra_state']['train_infos']
    report['checks']['checkpoint_extra_restart']=restarted['model']['_extra_state']['train_infos']
    assert report['checks']['checkpoint_extra_restart']['step']>report['checks']['checkpoint_extra_initial']['step']
    # The normal --finetune CLI must load these very same MACE-LES weights.
    fine=copy.deepcopy(config);fine['training']['numb_steps']=4;fine['training']['save_freq']=4
    run_cli(root/'finetune',fine,['--finetune',str(first_path)])
    final,final_model=checkpoint(root/'finetune/model.ckpt.pt')
    assert any(not torch.equal(final_model.state_dict()[n],v) for n,v in trained.state_dict().items() if v.is_floating_point())
    # Save/reload exact predictions, then export the trained original architecture.
    actual=final_model(**inputs)
    _,reloaded=checkpoint(root/'finetune/model.ckpt.pt')
    expected=reloaded(**inputs)
    report['checks']['reload_max_abs']={k:float((actual[k]-expected[k]).detach().abs().max()) for k in actual}
    for k in actual: torch.testing.assert_close(actual[k],expected[k],rtol=1e-10,atol=1e-10)
    from deepmd_demff.export import export_checkpoint
    manifest=export_checkpoint(root/'finetune/model.ckpt.pt',root/'trained.demff')
    from deepmd.infer import DeepPot
    dp=DeepPot(str(manifest),device='cuda',default_dtype='float64')
    tmap=dp.get_type_map();types=[tmap.index(s) for s in ['O','H','H','O','H','H']]
    exported=dp.eval(inputs['coord'].cpu().numpy(),inputs['box'].cpu().numpy(),types,fparam=inputs['fparam'].cpu().numpy())
    report['checks']['export_max_abs']={k:float(np.max(np.abs(v-actual[k].detach().cpu().numpy()))) for k,v in zip(['energy','force','virial'],exported)}
    for k,v in zip(['energy','force','virial'],exported):np.testing.assert_allclose(v,actual[k].detach().cpu().numpy(),atol=1e-8,rtol=1e-9)
    with (root/'exported-dp-test.log').open('w') as fp:
        result=subprocess.run([sys.executable,'-m','deepmd','test','-m',str(manifest),'-s',str(dataset/'valid'),'-n','2','-d',str(root/'exported-test')],stdout=fp,stderr=subprocess.STDOUT)
    assert result.returncode==0,(root/'exported-dp-test.log').read_text()
    for field in ['e','f','v']:
        assert np.isfinite(np.loadtxt(root/('exported-test.'+field+'.out'))).all()
    report['checks']['exported_standard_dp_test']={'exit_code':0,'frames':2,'scientific_acceptance':False}
    for stage in ['initial','restart','finetune']:
        curve=np.loadtxt(root/stage/'lcurve.out',ndmin=2);assert np.isfinite(curve).all()
        report['checks'][stage+'_lcurve']={'rows':len(curve),'first':curve[0].tolist(),'last':curve[-1].tolist()}
    report['status']='PASS';save();print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
