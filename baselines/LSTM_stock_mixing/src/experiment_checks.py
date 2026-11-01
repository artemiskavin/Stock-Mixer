"""Numerical compatibility checks and model-shape reports."""
import ast
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from model import get_loss
from experiment_io import atomic_json

OFFSETS = [0,127,255,383,511,639,739,755]


def relative_l2(value, reference):
    return ((value-reference).double().norm()/reference.double().norm().clamp_min(1e-12)).item()


def gradient(model, scale=1.):
    return torch.cat([p.grad.detach().reshape(-1)/scale for p in model.parameters() if p.grad is not None])


def inspect_or_check(args, model, batch_stream, get_batch, run_dir, metadata, arrays, project_root):
    original_text=(project_root/'reference/StockMixer/src/model.py').read_text()
    working_text=Path(__file__).with_name('model.py').read_text()
    def loss_ast(text):
        return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='get_loss'))
    assert loss_ast(original_text)==loss_ast(working_text)
    assert (project_root/'reference/StockMixer/src/evaluator.py').read_bytes()==Path(__file__).with_name('evaluator.py').read_bytes()
    for offset, batch in zip(OFFSETS,batch_stream(OFFSETS)):
        assert all(np.array_equal(t.cpu().numpy(), a) for t,a in zip(batch,get_batch(offset)))
    report={'configuration':metadata['configuration'],'parameter_count':sum(p.numel() for p in model.parameters()),
            'model_sha256':metadata['source_sha256']['model.py'],'author_loss_identical':True,
            'evaluator_identical':True,'delivery_equal':True,'trace':{},'checks':{}}
    handles=[]
    for name,module in model.named_modules():
        if name in ['conv','mixer','mixer.mix_layer','mixer.scale_mix_layer','channel_fc','stock_mixer','time_fc','time_fc_','rnn','head','stock_head']:
            def record(mod, inputs, output, key=name):
                tensor=output[0] if isinstance(output,tuple) else output
                report['trace'][key]={'shape':list(tensor.shape),'dtype':str(tensor.dtype)}
            handles.append(module.register_forward_hook(record))
    with torch.no_grad(): prediction=model(next(batch_stream([0]))[0])
    for handle in handles:handle.remove()
    assert prediction.shape==(1026,1) and torch.isfinite(prediction).all().item()
    if args.pilot_action=='inspect':
        atomic_json(run_dir/'inspection.json',report);print(json.dumps(report,indent=2),flush=True);return
    if args.reference_run is None:raise ValueError('FP32 reference snapshots required')
    reference=copy.deepcopy(model)

    def check_path(label, compiled, amp):
        candidate=torch.compile(model,backend='inductor',mode='default',dynamic=False) if compiled else model
        before_graphs = torch._dynamo.utils.counters['stats']['unique_graphs'] if compiled else 0
        results=[]
        for snapshot_file in [args.reference_run/'initial.pt',args.reference_run/'epochs/epoch_001.pt']:
            snapshot=torch.load(snapshot_file,map_location='cpu',weights_only=False)
            reference.load_state_dict(snapshot['model']);model.load_state_dict(snapshot['model'])
            for offset,batch in zip(OFFSETS,batch_stream(OFFSETS)):
                reference.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True)
                with torch.autocast('cuda',enabled=False):
                    expected=reference(batch[0]);ev=get_loss(expected,batch[3],batch[2],batch[1],1026,.1)
                ev[0].backward();eg=gradient(reference).clone()
                torch.cuda.synchronize();started=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.float16,enabled=amp):actual=candidate(batch[0])
                with torch.autocast('cuda',enabled=False):av=get_loss(actual,batch[3],batch[2],batch[1],1026,.1)
                scale=256. if amp else 1.;(av[0]*scale).backward();ag=gradient(model,scale)
                torch.cuda.synchronize();elapsed=time.perf_counter()-started
                valid=batch[1][:,0]>.5
                price_error=relative_l2(actual[valid],expected[valid]);return_error=relative_l2(av[3][valid],ev[3][valid])
                loss_error=abs(av[0].item()-ev[0].item())/max(abs(ev[0].item()),1e-12)
                grad_error=relative_l2(ag,eg)
                cosine=torch.nn.functional.cosine_similarity(ag.double()[None],eg.double()[None],eps=1e-12).item()
                finite=bool(torch.isfinite(actual).all().item() and torch.isfinite(av[0]).item() and torch.isfinite(ag).all().item())
                fp32=actual.dtype==torch.float32 and all(v.dtype==torch.float32 for v in av)
                passed=(finite and fp32 and price_error<=.01 and return_error<=.01 and loss_error<=.01 and grad_error<=.05 and cosine>=.99) if amp else (finite and fp32 and torch.allclose(actual,expected,atol=1e-5,rtol=1e-4) and loss_error<=1e-4 and grad_error<=1e-3)
                results.append({'snapshot':snapshot_file.name,'offset':offset,'finite':finite,'protected_fp32':fp32,
                                'price_relative_l2':price_error,'return_relative_l2':return_error,
                                'loss_relative_error':loss_error,'gradient_relative_l2':grad_error,
                                'gradient_cosine':cosine,'forward_backward_seconds':elapsed,'passed':bool(passed)})
                report['checks'][label]=results;atomic_json(run_dir/'preflight.json',report)
        if compiled:
            captured = torch._dynamo.utils.counters['stats']['unique_graphs'] - before_graphs
            report[label + '_compiled_graphs_added'] = captured
            if captured == 0:
                report['compile_fallback_reason'] = 'No optimizable graph captured for this model/path'
                return False
        return all(r['passed'] for r in results)

    if not check_path('eager_fp16',False,True):
        report.update(all_passed=False,selected_backend=None,reason='FP16 numerical validation failed.')
        atomic_json(run_dir/'preflight.json',report)
        raise FloatingPointError('Eager FP16 failed numerical validation.')
    selected='eager'
    if args.compile_policy=='auto':
        try:
            if check_path('compiled_fp32',True,False) and check_path('compiled_fp16',True,True):
                selected='compiled'
            else:report['compile_fallback_reason']='Compilation failed numerical validation.'
        except Exception as exc:
            report['compile_fallback_reason']=repr(exc)
    report.update(all_passed=True,selected_backend=selected,
                  compile_counters={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()})
    atomic_json(run_dir/'preflight.json',report)
    print(json.dumps({'configuration':metadata['configuration']['name'],'all_passed':True,
                      'selected_backend':selected,'parameter_count':report['parameter_count']}),flush=True)
