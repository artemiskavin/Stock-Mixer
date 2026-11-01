"""Atomic experiment logging, checkpointing, and RNG-state helpers."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import msvcrt

import numpy as np
import torch


def acquire_run_lock(run_dir):
    handle = (run_dir / 'active.lock').open('a+b')
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b'0'); handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        raise RuntimeError('Another process already owns this run directory')
    return handle  # Hold open for the lifetime of the process.


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def atomic_save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def rng_state():
    return {'numpy': np.random.get_state(), 'python': random.getstate(),
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    np.random.set_state(state['numpy']); random.setstate(state['python'])
    torch.set_rng_state(state['torch_cpu']); torch.cuda.set_rng_state_all(state['torch_cuda'])


def parameter_hash(model):
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                 for p in model.state_dict().values())).hexdigest()


def cpu_weights(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def select_epochs(rows):
    """Select epochs by validation loss and IC with deterministic tie-breaking."""
    if not rows or any(not np.isfinite(r['validation_loss']) or
                       not np.isfinite(r['validation_IC']) for r in rows):
        raise ValueError('Selection requires finite validation loss and IC for every epoch')
    best_loss = min(rows, key=lambda r: (r['validation_loss'], r['epoch']))
    best_ic = min(rows, key=lambda r: (-r['validation_IC'], r['validation_loss'], r['epoch']))
    threshold = 1.10 * best_loss['validation_loss']
    eligible = [r for r in rows if r['validation_loss'] <= threshold]
    primary = min(eligible, key=lambda r: (-r['validation_IC'], r['validation_loss'], r['epoch']))
    return {'minimum_validation_loss': best_loss['validation_loss'], 'loss_tolerance': .10,
            'eligibility_threshold': threshold, 'eligible_epochs': [r['epoch'] for r in eligible],
            'primary_epoch': primary['epoch'], 'secondary_epoch': best_loss['epoch'],
            'unrestricted_best_ic_epoch': best_ic['epoch']}


def finalize_checkpoints(run_dir, rows):
    selection = select_epochs(rows)
    mapping = {'best_validation_loss.pt': selection['secondary_epoch'],
               'best_validation_ic_unrestricted.pt': selection['unrestricted_best_ic_epoch'],
               'tolerance_selected.pt': selection['primary_epoch'], 'last_epoch.pt': rows[-1]['epoch']}
    for name, epoch in mapping.items():
        shutil.copyfile(run_dir / 'epochs' / ('epoch_%03d.pt' % epoch), run_dir / name)
    atomic_json(run_dir / 'selection.json', selection)
    return selection
