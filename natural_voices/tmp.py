import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torchaudio import save as save_audio

import utils
from data_utils import (
  TextAudioLoader,
  TextAudioCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)

from train import train_and_evaluate
# from text.symbols import symbols
import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


torch.backends.cudnn.benchmark = True
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '0'

  hps = utils.get_hparams()
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):

  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioCollate()
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
  
  net_g = SynthesizerTrn(
      len(train_dataset.text_mapper.symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank])
  net_d = DDP(net_d, device_ids=[rank])

  # Load checkpoints to finetune
  _, _, _, _ = utils.load_checkpoint(utils.latest_checkpoint_path(hps.finetune.ckpt_path, "G_*.pth"), net_g, optim_g)
  _, _, _, _ = utils.load_checkpoint(utils.latest_checkpoint_path(hps.finetune.ckpt_path, "D_*.pth"), net_d, optim_d)
  epoch_str = 1
  global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
      x, x_lengths = x.cuda(0), x_lengths.cuda(0)
      spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
      y, y_lengths = y.cuda(0), y_lengths.cuda(0)

      # remove else
      x = x[:1]
      x_lengths = x_lengths[:1]
      spec = spec[:1]
      spec_lengths = spec_lengths[:1]
      y = y[:1]
      y_lengths = y_lengths[:1]
      break

  y_hat, attn, mask, *_ = net_g.module.infer(x, x_lengths, max_len=1000)
  y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length
  print(y_hat.size())

  out_path = os.path.join(writer_eval.log_dir, "gen_%05d.wav"%global_step)
  save_audio(out_path, y_hat[0].cpu() * hps.data.max_wav_value, hps.data.sampling_rate)

  out_path = os.path.join(writer_eval.log_dir, "gt_%05d.wav"%global_step)
  save_audio(out_path, y[0].cpu() * hps.data.max_wav_value, hps.data.sampling_rate)

       
if __name__ == "__main__":
  main()
