from typing import Optional, Union
import json, os
import random

import numpy as np
import torch
import torch.distributed as dist

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from datalibs.dataset import FFHQDataset
from datalibs.compose import Resize, Normalization, Frequency_Filtering, SourceResolution


class BaseTrainer():
    def __init__(self, ckpt_dir: str, log_dir: str, batch_size: int, num_workers: int = 6):
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def set_seed(self, seed: int, deterministic: bool = False):
        """Seed every RNG we touch.

        `deterministic=False` by default: cuDNN autotuning is worth a large
        throughput win and the diffusion objective is stochastic anyway, so
        bitwise determinism buys nothing here. Evaluation seeds its own
        generators explicitly, which is what actually has to be reproducible.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

    # kept for callers that still use the old spelling
    set_seeed = set_seed

    def __checkDirectory__(self):
        for d in (self.config.trainer.ckpt_dir,
                  os.path.join(self.config.trainer.ckpt_dir, self.config.model.name),
                  self.config.trainer.log_dir):
            os.makedirs(d, exist_ok=True)

    def initialize(self, gpu, size):
        """Bind this process to its GPU, and only start a process group for real
        multi-process runs -- a single-GPU job does not need one."""
        torch.cuda.set_device(gpu)
        if size > 1 and not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ["LOCAL_RANK"] = str(gpu)
            dist.init_process_group(backend="nccl", rank=gpu, world_size=size)

    def makeDatasets(self,
                     data_path: str,
                     img_dir: str,
                     frequency_rate: Union[str, float] = "random",
                     mode: str = "high_pass",
                     img_size: int = 512,
                     source_size: Optional[int] = None,
                     rate_range=(0.0, 1.0),
                     rate_sampler: str = "uniform",
                     frequency_img: str = "gray",
                     mean: Optional[Union[list, float]] = 0.5,
                     std: Optional[Union[list, float]] = 0.5,
                     caption_path: Optional[str] = None,
                     ddp=False):
        """Build train/valid/test loaders.

        Train uses the randomised cutoff (`frequency_rate='random'`); valid and
        test are always driven at explicit fixed cutoffs by the evaluator, so
        their transform is built with whatever `frequency_rate` is passed in and
        the evaluator rebuilds it per operating point.
        """
        def build(rate):
            steps = [Resize((img_size, img_size))]
            if source_size:
                steps.append(SourceResolution(source_size))
            return transforms.Compose(steps + [
                Normalization(mean, std),
                Frequency_Filtering(frequency_img=frequency_img,
                                    frequency_rate=rate,
                                    mode=mode,
                                    rate_range=rate_range,
                                    rate_sampler=rate_sampler,
                                    out_range="model"),
            ])

        self.transform_train = build(frequency_rate)
        self.transform_valid = build(frequency_rate)
        self.transform_test = build(frequency_rate)
        # old misspelled attribute names, kept so existing call sites keep working
        self.trasnform_train = self.transform_train
        self.trasnform_valid = self.transform_valid
        self.trasnform_test = self.transform_test

        common = dict(data_dir=data_path, img_dir=img_dir, caption_path=caption_path)
        self.train_dataset = FFHQDataset(mode="train", transforms=self.transform_train, **common)
        self.valid_dataset = FFHQDataset(mode="valid", transforms=self.transform_valid, **common)
        self.test_dataset = FFHQDataset(mode="test", transforms=self.transform_test, **common)

        loader_kw = dict(batch_size=self.batch_size, num_workers=self.num_workers,
                         pin_memory=True, drop_last=False)
        if self.num_workers > 0:
            loader_kw.update(persistent_workers=True, prefetch_factor=4)

        if ddp:
            self.train_dataset_sampler = DistributedSampler(self.train_dataset, drop_last=True)
            self.valid_dataset_sampler = DistributedSampler(self.valid_dataset, drop_last=True)
            self.test_dataset_sampler = DistributedSampler(self.test_dataset, drop_last=True)
            self.loader_train = DataLoader(self.train_dataset, shuffle=False,
                                           sampler=self.train_dataset_sampler, **loader_kw)
            self.loader_valid = DataLoader(self.valid_dataset, shuffle=False,
                                           sampler=self.valid_dataset_sampler, **loader_kw)
            self.loader_test = DataLoader(self.test_dataset, shuffle=False,
                                          sampler=self.test_dataset_sampler, **loader_kw)
        else:
            self.loader_train = DataLoader(self.train_dataset, shuffle=True, **loader_kw)
            self.loader_valid = DataLoader(self.valid_dataset, shuffle=False, **loader_kw)
            self.loader_test = DataLoader(self.test_dataset, shuffle=False, **loader_kw)

    def makeTensorBoard(self):
        self.summaryWriter = SummaryWriter(log_dir=self.log_dir)

    def json_load(self, path: str):
        with open(path, "r") as jsonFile:
            return json.load(jsonFile)
