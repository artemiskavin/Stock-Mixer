"""Author daily-window trainer with recorded experiments and 100-epoch selection."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import random
import shutil
import subprocess
import time
import numpy as np
import torch
from load_data import load_EOD_data
from evaluator import evaluate
from model import get_loss, LSTMStockForecaster
from experiment_io import (atomic_json, atomic_save, rng_state, restore_rng,
                           parameter_hash, cpu_weights, finalize_checkpoints, select_epochs, acquire_run_lock)

parser = argparse.ArgumentParser(description=__doc__)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument('--run', action='store_true')
mode.add_argument('--pilot', action='store_true')
parser.add_argument('--pilot-action', choices=['inspect', 'check', 'time'], default='time')
parser.add_argument('--pilot-epochs', type=int, choices=range(1, 5), default=1)
parser.add_argument('--seed-index', type=int, choices=[0, 1, 2], default=0)
parser.add_argument('--data-mode', choices=['cpu_sync', 'cpu_async', 'gpu_preload'], default='gpu_preload')
parser.add_argument('--diagnostics', choices=['original', 'batched'], default='batched')
parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp16')
parser.add_argument('--compile-policy', choices=['auto', 'eager'], default='auto')
parser.add_argument('--reference-run', type=Path)
parser.add_argument('--preflight', type=Path)
parser.add_argument('--resume', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
setup_started = time.perf_counter()
project_root = next(p for p in Path(__file__).resolve().parents
                    if (p / 'reference/StockMixer').is_dir() and (p / 'docs/protocol.md').is_file())
configuration = json.loads(Path(__file__).with_name('experiment.json').read_text())
run_dir = args.output.resolve()
allowed = project_root / 'artifacts' / ('runs' if args.run else 'preflight')
if not run_dir.is_relative_to(allowed):
    raise ValueError('Output directory is outside the designated artifact area')
if args.run and (args.precision != 'fp16' or args.preflight is None):
    raise ValueError('Training requires FP16 and a valid numerical preflight.')
if args.resume:
    if args.resume.resolve() != run_dir / 'last_training_state.pt':
        raise ValueError('Resume must use this incomplete run last training state')
    if json.loads((run_dir / 'status.json').read_text())['status'] == 'completed':
        raise ValueError('Completed runs are immutable')
else:
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'epochs').mkdir()
run_lock = acquire_run_lock(run_dir)
SEED_PAIRS = [(123456789, 12345678), (234567890, 23456789), (345678901, 34567890)]
numpy_seed, torch_seed = SEED_PAIRS[args.seed_index]
np.random.seed(numpy_seed)
torch.random.manual_seed(torch_seed)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU required; no automatic CPU precision fallback')
device = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
data_path = '../dataset'
market_name = 'NASDAQ'
stock_num = 1026
lookback_length = 16
epochs = 100 if args.run else args.pilot_epochs
valid_index = 756
test_index = 1008
fea_num = 5
market_num = 20
steps = 1
learning_rate = .001
alpha = .1
scale_factor = 3
dataset_path = '../dataset/NASDAQ'
with open(os.path.join(dataset_path, 'eod_data.pkl'), 'rb') as f: eod_data = pickle.load(f)
with open(os.path.join(dataset_path, 'mask_data.pkl'), 'rb') as f: mask_data = pickle.load(f)
with open(os.path.join(dataset_path, 'gt_data.pkl'), 'rb') as f: gt_data = pickle.load(f)
with open(os.path.join(dataset_path, 'price_data.pkl'), 'rb') as f: price_data = pickle.load(f)
trade_dates = mask_data.shape[1]
assert eod_data.shape == (1026, 1245, 5) and trade_dates == 1245
model = LSTMStockForecaster(input_size=fea_num, hidden_size=32, stocks=stock_num, market=market_num).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(.9, .999),
                             eps=1e-8, weight_decay=0, amsgrad=False, foreach=False, fused=False)
batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)
use_amp = args.precision == 'fp16'
scaler = torch.amp.GradScaler('cuda', enabled=use_amp, init_scale=256,
                            growth_factor=2, backoff_factor=.5, growth_interval=74001)
model_runner = model
gpu_arrays = None
if args.data_mode == 'gpu_preload':
    gpu_arrays = tuple(torch.from_numpy(a).to(device) for a in (eod_data, mask_data, price_data, gt_data))
source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in Path(__file__).parent.glob('*.py')}
source_hashes['experiment.json'] = hashlib.sha256(Path(__file__).with_name('experiment.json').read_bytes()).hexdigest()
dataset_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in Path(dataset_path).glob('*.pkl')}
initial_hash = parameter_hash(model)
metadata = {'scope': 'scientific 100-epoch run' if args.run else 'implementation preflight',
            'configuration': configuration, 'numpy_seed': numpy_seed, 'torch_seed': torch_seed,
            'seed_index': args.seed_index, 'epochs': epochs, 'parameter_count': sum(p.numel() for p in model.parameters()),
            'initial_weights_sha256': initial_hash, 'source_sha256': source_hashes,
            'dataset_sha256': dataset_hashes, 'python': platform.python_version(),
            'torch': torch.__version__, 'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(),
            'tf32_matmul': False, 'tf32_cudnn': False, 'cudnn_benchmark': False,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'torch_threads': torch.get_num_threads(),
            'scaler_configuration': {'init_scale': 256, 'growth_factor': 2, 'backoff_factor': .5, 'growth_interval': 74001},
            'scientific_settings': {'stocks': 1026, 'features': 5, 'lookback': 16, 'market': 20,
                                    'validation_boundary': 756, 'test_boundary': 1008, 'alpha': .1,
                                    'learning_rate': .001, 'updates_per_epoch': 740, 'eligible_offsets': 756,
                                    'adam_betas': [.9, .999], 'adam_eps': 1e-8, 'weight_decay': 0,
                                    'amsgrad': False, 'foreach': False, 'fused': False, 'early_stopping': False},
            'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'compile_environment': {k: os.environ.get(k) for k in ['TEMP','TMP','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR']},
            'torch_build': torch.__config__.show(),
            'driver': subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()}
assert metadata['parameter_count'] == configuration['expected_parameters']
if not args.resume: atomic_json(run_dir / 'metadata.json', metadata)


def forward_loss(batch):
    data_batch, mask_batch, price_batch, gt_batch = batch
    with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
        prediction = model_runner(data_batch)
    with torch.autocast('cuda', enabled=False):
        result = get_loss(prediction, gt_batch, price_batch, mask_batch, stock_num, alpha)
    return prediction, result

def validate(start_index, end_index):
    with torch.no_grad():
        count = end_index - start_index
        offsets = range(start_index - lookback_length - steps + 1,
                        end_index - lookback_length - steps + 1)
        if args.diagnostics == 'original':
            arrays = np.zeros((3, stock_num, count), dtype=float)
            totals = [0., 0., 0.]
        else:
            arrays = torch.empty((3, stock_num, count), device=device, dtype=torch.float32)
            diagnostics = torch.empty((count, 3), device=device, dtype=torch.float32)
        finite = torch.empty(count, device=device, dtype=torch.bool)
        for day, batch in enumerate(batch_stream(offsets)):
            prediction, values = forward_loss(batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = values
            finite[day] = torch.isfinite(prediction).all() & torch.isfinite(cur_loss)
            if args.diagnostics == 'original':
                for k, value in enumerate(values[:3]):
                    totals[k] += value.item()
                for k, value in enumerate((cur_rr, batch[3], batch[1])):
                    arrays[k, :, day] = value[:, 0].cpu()
            else:
                diagnostics[day] = torch.stack([v.detach() for v in values[:3]])
                for k, value in enumerate((cur_rr, batch[3], batch[1])):
                    arrays[k, :, day] = value[:, 0]
        if not finite.all().item():
            raise FloatingPointError('Nonfinite evaluation prediction or loss')
        if args.diagnostics == 'batched':
            # Each FP32 scalar becomes a Python float, then is accumulated in
            # exactly the same day order as the released .item() loop.
            rows = diagnostics.cpu().tolist()
            totals = [sum(row[k] for row in rows) for k in range(3)]
            arrays = arrays.cpu().numpy().astype(float)
        cur_valid_perf = evaluate(*arrays)
    return *(value / count for value in totals), cur_valid_perf

def get_batch(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = lookback_length
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        eod_data[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1))

def batch_stream(offsets):
    """Yield training windows in the supplied offset order."""
    if args.data_mode == 'cpu_sync':
        for offset in offsets:
            yield tuple(torch.Tensor(x).to(device) for x in get_batch(int(offset)))
    elif args.data_mode == 'gpu_preload':
        eod, mask, price, gt = gpu_arrays
        for offset in offsets:
            offset = int(offset)
            yield (eod[:, offset:offset + lookback_length, :].contiguous(),
                   mask[:, offset:offset + lookback_length + steps].amin(1, keepdim=True),
                   price[:, offset + lookback_length - 1:offset + lookback_length].contiguous(),
                   gt[:, offset + lookback_length + steps - 1:offset + lookback_length + steps].contiguous())
    else:
        # Two reusable host/device slots. Protect host reuse with copy completion
        # and device reuse with compute completion; neither changes offset order.
        offsets = list(offsets)
        sample = get_batch(int(offsets[0]))
        host = [[torch.empty(x.shape, dtype=torch.float32, pin_memory=True) for x in sample]
                for _ in range(2)]
        buffers = [[torch.empty(x.shape, dtype=torch.float32, device=device) for x in sample]
                   for _ in range(2)]
        copy_stream = torch.cuda.Stream()
        ready = [torch.cuda.Event() for _ in range(2)]
        consumed = [torch.cuda.Event() for _ in range(2)]
        used = [False, False]
        for event in consumed:
            event.record()

        def enqueue(index):
            slot = index % 2
            if used[slot]:
                ready[slot].synchronize()
            for target, array in zip(host[slot], get_batch(int(offsets[index]))):
                target.copy_(torch.from_numpy(array))
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(consumed[slot])
                for target, source in zip(buffers[slot], host[slot]):
                    target.copy_(source, non_blocking=True)
                ready[slot].record(copy_stream)
            used[slot] = True

        for index in range(min(2, len(offsets))):
            enqueue(index)
        for index in range(len(offsets)):
            slot = index % 2
            torch.cuda.current_stream().wait_event(ready[slot])
            yield tuple(buffers[slot])
            consumed[slot].record()
            if index + 2 < len(offsets):
                enqueue(index + 2)


def model_checkpoint(epoch, record):
    return {'epoch': epoch, 'model': cpu_weights(model), 'configuration': metadata,
            'metrics': dict(record)}


def warm_runner(backend):
    global model_runner
    saved_rng = rng_state()
    saved_weights = cpu_weights(model)
    started = time.perf_counter()
    try:
        model_runner = torch.compile(model, backend='inductor', mode='default', dynamic=False) if backend == 'compiled' else model
        batch = next(batch_stream([0]))
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            prediction, values = forward_loss(batch)
            (values[0] * (256. if use_amp else 1.)).backward()
            flat = torch.cat([p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None])
            if not (torch.isfinite(prediction).all() & torch.isfinite(values[0]) & torch.isfinite(flat).all()).item():
                raise FloatingPointError('Nonfinite values detected during warm-up.')
        with torch.no_grad(): forward_loss(batch)
        torch.cuda.synchronize()
    finally:
        model.load_state_dict(saved_weights)
        model.zero_grad(set_to_none=True)
        restore_rng(saved_rng)
    assert parameter_hash(model) == initial_hash or args.resume
    return time.perf_counter() - started


def run_training():
    global model_runner
    backend = 'eager'
    if args.run:
        passed = json.loads(args.preflight.read_text())
        if not passed['all_passed'] or passed['model_sha256'] != source_hashes['model.py']:
            raise ValueError('Missing or stale numerical preflight')
        backend = passed['selected_backend']
    elif args.compile_policy == 'auto':
        backend = 'compiled'
    if args.compile_policy == 'eager': backend = 'eager'
    rows = []
    start_epoch = 0
    saved = None
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['configuration']['source_sha256'] != source_hashes or saved['configuration']['dataset_sha256'] != dataset_hashes:
            raise ValueError('Source or dataset changed since last training state')
        if saved['configuration']['seed_index'] != args.seed_index:
            raise ValueError('Resume seed mismatch')
        for key in ['data_mode','precision','compile_policy']:
            if saved['configuration']['arguments'][key] != getattr(args,key):
                raise ValueError('Resume execution setting changed: '+key)
        for key in ['torch','numpy','cuda_runtime','cudnn','driver','tf32_matmul','tf32_cudnn','cudnn_benchmark','deterministic_algorithms','cudnn_deterministic']:
            if saved['configuration'][key] != metadata[key]:
                raise ValueError('Resume environment/backend changed: '+key)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['grad_scaler'])
        batch_offsets[:] = saved['batch_offsets']; rows = saved['rows']; start_epoch = saved['epoch']
        # Preserve committed timing values from the final CSV flush; the last
        # training-state payload is written just before that timing is known.
        if (run_dir/'per_epoch_metrics.csv').exists():
            with (run_dir/'per_epoch_metrics.csv').open(newline='') as f:
                committed_times={int(r['epoch']):float(r['complete_epoch_seconds']) for r in csv.DictReader(f)}
            for row in rows:
                if row['epoch'] in committed_times: row['complete_epoch_seconds']=committed_times[row['epoch']]

    try:
        metadata['warmup_seconds'] = warm_runner(backend)
    except Exception as exc:
        if backend != 'compiled' or isinstance(exc, FloatingPointError): raise
        metadata['compile_fallback_reason'] = repr(exc)
        backend = 'eager'
        metadata['warmup_seconds'] = warm_runner(backend)
    metadata['selected_backend'] = backend
    metadata['compile_counters_after_warmup'] = {str(k): dict(v) for k,v in torch._dynamo.utils.counters.items()}
    if saved:
        restore_rng(saved['rng'])
        for name, p in model.named_parameters():
            p.grad = saved['gradients'][name].to(device) if name in saved['gradients'] else None
    atomic_json(run_dir / 'metadata.json', metadata)
    atomic_json(run_dir / 'status.json', {'status': 'running', 'completed_epoch': start_epoch,
                                         'backend': backend, 'pid': os.getpid()})
    if not saved: atomic_save(run_dir / 'initial.pt', model_checkpoint(0, {}))
    torch.cuda.synchronize()
    metadata['setup_seconds'] = time.perf_counter() - setup_started
    count = valid_index - lookback_length - steps + 1
    for epoch in range(start_epoch, epochs):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        np.random.shuffle(batch_offsets)
        permutation_hash = hashlib.sha256(batch_offsets.tobytes()).hexdigest()
        diagnostics = torch.empty((count, 3), device=device, dtype=torch.float32)
        finite = torch.empty(count, device=device, dtype=torch.bool)
        totals = [0., 0., 0.]
        for j, batch in enumerate(batch_stream(batch_offsets[:count])):
            optimizer.zero_grad(set_to_none=False)
            prediction, values = forward_loss(batch)
            scaler.scale(values[0]).backward()
            if use_amp: scaler.unscale_(optimizer)
            gradients = torch.cat([p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None])
            finite[j] = torch.isfinite(prediction).all() & torch.isfinite(values[0]) & torch.isfinite(gradients).all()
            scaler.step(optimizer); scaler.update()
            if args.diagnostics == 'original':
                for k,v in enumerate(values[:3]): totals[k] += v.item()
            else: diagnostics[j] = torch.stack([v.detach() for v in values[:3]])
            if (j+1)%32 == 0 or j+1 == count:
                if not finite[max(0,j-31):j+1].all().item():
                    raise FloatingPointError('Nonfinite values or a skipped optimizer update detected.')
        if args.diagnostics == 'batched':
            detached_rows = diagnostics.cpu().tolist()
            totals = [sum(row[k] for row in detached_rows) for k in range(3)]
        train_seconds = time.perf_counter() - started
        evaluation_started = time.perf_counter()
        val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
        test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
        if not np.isfinite(val_perf['IC']): raise FloatingPointError('Nonfinite validation IC; selection undefined')
        record = {'epoch': epoch+1, 'training_updates': count, 'train_loss': totals[0]/count,
                  'train_mse_component': totals[1]/count, 'train_rank_unweighted': totals[2]/count,
                  'validation_loss': val_loss, 'validation_mse_component': val_reg_loss,
                  'validation_rank_unweighted': val_rank_loss, 'test_loss': test_loss,
                  'test_mse_component': test_reg_loss, 'test_rank_unweighted': test_rank_loss,
                  **{'validation_'+k:float(v) for k,v in val_perf.items()},
                  **{'test_'+k:float(v) for k,v in test_perf.items()},
                  'permutation_sha256': permutation_hash, 'train_seconds': train_seconds,
                  'evaluation_seconds': time.perf_counter()-evaluation_started,
                  'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
                  'peak_reserved_MiB': torch.cuda.max_memory_reserved()/2**20,
                  'scaler_scale': scaler.get_scale(), 'complete_epoch_seconds': 0.}
        rows.append(record)
        checkpoint_path = run_dir / 'epochs' / ('epoch_%03d.pt' % (epoch+1))
        atomic_save(checkpoint_path, model_checkpoint(epoch+1, record))
        current = select_epochs(rows)
        if current['secondary_epoch'] == epoch+1: shutil.copyfile(checkpoint_path, run_dir/'best_validation_loss.pt')
        if current['unrestricted_best_ic_epoch'] == epoch+1: shutil.copyfile(checkpoint_path, run_dir/'best_validation_ic_unrestricted.pt')
        shutil.copyfile(checkpoint_path, run_dir/'last_epoch.pt')
        atomic_save(run_dir/'last_training_state.pt',
                    {'epoch': epoch+1, 'model': cpu_weights(model), 'optimizer': optimizer.state_dict(),
                     'grad_scaler': scaler.state_dict(), 'rng': rng_state(), 'batch_offsets': batch_offsets.copy(),
                     'gradients': {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None},
                     'rows': rows, 'configuration': metadata})
        torch.cuda.synchronize(); record['complete_epoch_seconds'] = time.perf_counter()-started
        temp_csv = run_dir/'per_epoch_metrics.csv.tmp'
        with temp_csv.open('w', newline='', encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(record)); writer.writeheader(); writer.writerows(rows)
        os.replace(temp_csv,run_dir/'per_epoch_metrics.csv')
        atomic_json(run_dir/'status.json', {'status':'running','completed_epoch':epoch+1,
                                           'backend':backend,'pid':os.getpid(),
                                           'last_epoch_seconds':record['complete_epoch_seconds']})
        print(json.dumps({k:record[k] for k in ['epoch','complete_epoch_seconds','peak_allocated_MiB']}),flush=True)
    if args.run and len(rows) != 100: raise RuntimeError('Scientific run did not complete all 100 epochs')
    selection = finalize_checkpoints(run_dir,rows)
    metadata['compile_counters_final'] = {str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()}
    metadata['total_training_seconds'] = sum(r['complete_epoch_seconds'] for r in rows)
    atomic_json(run_dir/'metadata.json',metadata)
    atomic_json(run_dir/'status.json',{'status':'completed','completed_epoch':epochs,
                                      'scope':metadata['scope'],'selection':selection,'backend':backend})


if __name__ == '__main__':
    try:
        if args.pilot and args.pilot_action in ['inspect','check']:
            from experiment_checks import inspect_or_check
            inspect_or_check(args,model,batch_stream,get_batch,run_dir,metadata,
                             (eod_data,mask_data,price_data,gt_data),project_root)
        else: run_training()
    except Exception as exc:
        atomic_json(run_dir/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
        atomic_json(run_dir/'status.json',{'status':'failed','message':str(exc),'pid':os.getpid()})
        raise
