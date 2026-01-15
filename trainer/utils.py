import os
import sys
import atexit
import logging
import datetime
import numpy as np

from typing import Dict, Any, Optional
from typing_extensions import override

import torch
import torch.nn as nn

import torch.distributed as dist

import matplotlib.pyplot as plt

from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
from torch.distributed import init_process_group, new_group


from libs.layers import LinearWithLoRA, CrossAttentionWithLoRA


from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.rank_zero import rank_zero_info
from lightning.fabric.plugins import ClusterEnvironment

from lightning.fabric.utilities.distributed import (
    _distributed_is_initialized,
    _get_default_process_group_backend_for_device,
    # _init_dist_connection,
    _sync_ddp_if_available,
)

log = logging.getLogger(__name__)

def _destroy_dist_connection() -> None:
    if _distributed_is_initialized():
        torch.distributed.destroy_process_group()


# class CustomDDPStrategy(FSDPStrategy):

class CustomDDPStrategy(DDPStrategy):

    def __init__(self, *args, **kwargs):
        self.sharedFilePath = kwargs["sharedFilePath"]
        del kwargs["sharedFilePath"]
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        # self.kwargs = kwargs

    @property
    @override
    def distributed_sampler_kwargs(self) -> Dict:
        return {"num_replicas": (self.num_nodes * self.num_processes), "rank": self.global_rank, "drop_last":True}
    
    def _init_dist_connection(
        self,
        cluster_environment: "ClusterEnvironment",
        torch_distributed_backend: str,
        global_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        init_method = 'file://./sharedfile',
        **kwargs: Any,
    ) -> None:
        """Utility function to initialize distributed connection by setting env variables and initializing the distributed
        process group.

        Args:
            cluster_environment: ``ClusterEnvironment`` instance
            torch_distributed_backend: Backend to use (includes `nccl` and `gloo`)
            global_rank: Rank of the current process
            world_size: Number of processes in the group
            kwargs: Kwargs for ``init_process_group``

        Raises:
            RuntimeError:
                If ``torch.distributed`` is not available

        """
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available. Cannot initialize distributed process group")
        if torch.distributed.is_initialized():
            log.debug("torch.distributed is already initialized. Exiting early")
            return
        global_rank = global_rank if global_rank is not None else cluster_environment.global_rank()
        world_size = world_size if world_size is not None else cluster_environment.world_size()
        os.environ["MASTER_ADDR"] = cluster_environment.main_address
        os.environ["MASTER_PORT"] = str(cluster_environment.main_port)
        os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
        os.environ['NCCL_IB_DISABLE']= '1'
        # os.environ['NCCL_DEBUG'] = 'INFO'
        # os.environ['NCCL_P2P_LEVEL'] = 'NVL'
        os.environ['PYTHONFAULTHANDLER'] = '1'
        log.info(f"Initializing distributed: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
        log.info(f"InitMethod : {init_method}")
        torch.cuda.set_device(global_rank)
        torch.distributed.init_process_group(torch_distributed_backend, rank=global_rank, world_size=world_size, init_method=init_method, **kwargs)

        if torch_distributed_backend == "nccl":
            # PyTorch >= 2.4 warns about undestroyed NCCL process group, so we need to do it at program exit
            atexit.register(_destroy_dist_connection)

        # On rank=0 let everyone know training is starting
        rank_zero_info(
            f"{'-' * 100}\n"
            f"distributed_backend={torch_distributed_backend}\n"
            f"All distributed processes registered. Starting with {world_size} processes\n"
            f"{'-' * 100}\n"
        )

    def setup_distributed(self) -> None:
        log.debug(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()
        self.set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        self._init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout, init_method='file://' + self.sharedFilePath+ '/sharedfile')
    
    # # For ddp
    # def setup_enviroment(self)->None:
    #     self.setup_distributed()

    #  FOR fsdp
    @override
    def setup_environment(self) -> None:
        # super().setup_environment()
        log.debug(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()

        # determine which process we are and world size
        self.set_world_ranks()

        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        self._init_dist_connection(self.cluster_environment, self._process_group_backend, 
                                   timeout=self._timeout,
                                   init_method = 'file://' + self.sharedFilePath+ '/sharedfile')

        # if 'device_mesh' in the `kwargs` is provided as a tuple, update it into the `DeviceMesh` object here
        if isinstance(self.kwargs.get("device_mesh"), tuple):
            from torch.distributed.device_mesh import init_device_mesh

            self.kwargs["device_mesh"] = init_device_mesh("cuda", self.kwargs["device_mesh"])




def save(ckpt_dir, net, optim, epoch, model_name="PatchPainting"):
    r"""
    Model Saver

    Inputs:
        ckpt_dir   : (string) check point directory
        netG       : (nn.module) Generator Network
        opitmG     : (torch.optim) Generator's Optimizers
        epoch      : (int) Now Epoch
        model_name : (string) Saving model file's name
    """
    if hasattr(net, "module"):
        netG_dicts = net.module.state_dict()
        try:
            optimG_dicts = optim.module.state_dict()
        except:
            optimG_dicts = optim.state_dict()
    else:
        netG_dicts = net.state_dict()
        optimG_dicts = optim.state_dict()

    torch.save({"net": netG_dicts,
                "optim" : optimG_dicts},
                os.path.join(ckpt_dir, model_name, f"{model_name}_{epoch}.pth"))

def load_gen(ckpt_dir,  netG,  optimG, name, epoch=None, gpu=None):
    r"""
    Model Lodaer

    Inputs:
        ckpt_dir : (string) check point directory
        netG     : (nn.module) Generator Network
        opitmG   : (torch.optim) Generator's Optimizers
        step     : (int) find step.  if NOne, last scale

    """
    ckpt_lst = os.listdir(ckpt_dir)

    if epoch is not None:
        ckptFile = os.path.join(ckpt_dir, name+f"_{epoch}.pth")
    else:
        ckpt_lst.sort()
        ckptFile = os.path.join(ckpt_dir, ckpt_lst[-1])

    if not os.path.exists(ckptFile):
        raise ValueError(f"Please Check Check Point File Path or Epoch, File is not exists!")

    # Load Epochs Now
    epoch = int(ckpt_lst[-1].split("_")[-1][:-4])

    # Load Model 
    if gpu is not None:
        dist.barrier()
        mapLocation = {"cuda:0": f"cuda:{gpu}"}
        dict_model = torch.load(ckptFile, map_location=mapLocation)
    else:
        dict_model = torch.load(ckptFile)
    
    try:
        netG.load_state_dict(dict_model['netG'])
    except:
        netG.module.load_state_dict(dict_model['netG'])

    optimG.load_state_dict(dict_model["optimG"])

    return netG,  optimG, epoch



def plot_recon_figures(sample, pred, output_path, step, num_figures = 5, save=False):

    fig, axs = plt.subplots(num_figures, 3, figsize=(30,15))
    fig.tight_layout()
    axs[0,0].set_title('Ground-truth')
    axs[0,1].set_title('Reconstruction')
    axs[0,2].set_title('Comparison')

    for ax, s, p in zip(axs, sample, pred):

        # cor = np.corrcoef([pred, sample])[0,1]

        x_axis = np.arange(0, sample.shape[1])
        # groundtruth
        ax[0].plot(x_axis, s)
       
        # pred
        ax[1].plot(x_axis, p)
        # ax[1].set_ylabel('cor: %.4f'%cor, weight = 'bold')
        # ax[1].yaxis.set_label_position("right")

        ax[2].plot(x_axis, s, "b")
        ax[2].plot(x_axis, p, "r")
    
    if save:
        fig_name = f'reconst-{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}-{step}.png'
        fig.savefig(os.path.join(output_path, fig_name))
    plt.close(fig)
    return fig


def plot_recon_figures_bychannels(sample, pred,  output_path, step, save=True):

    for s, p in zip(sample, pred):
        fig, axs = plt.subplots(1, 3, figsize=(15,12))
        fig.tight_layout()
        axs[0,0].set_title('Ground-truth')
        axs[0,1].set_title('Reconstruction')
        axs[0,2].set_title('Comparison')
        if save:
            fig_name = f'reconst-{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}-{step}.png'
            fig.savefig(os.path.join(output_path, fig_name))
        plt.close(fig)

def add_lora_to_model(model, rank=4, alpha=1.0):
    modules_to_replace = []

    for name, module in model.named_modules():
        # if isinstance(module, nn.Linear):
        #     modules_to_replace.append((name, module))
        if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
            modules_to_replace.append((name, module))

    for name, module in modules_to_replace:
        # if isinstance(module, nn.Linear):
        #     setattr(model, name, LinearWithLoRA(module, rank, alpha))
        if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
            setattr(model, name, CrossAttentionWithLoRA(module, rank, alpha))
    return model