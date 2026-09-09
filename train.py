import warnings
import logging
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")

import argparse
import torch, os
import torch.nn as nn

# from trainer.eeg_ldm2_trainer import EEGLDM2Trainer as trainer

os.environ.setdefault("HF_HOME", "/home/work/data/hf_cache")

parser = argparse.ArgumentParser(description="Evaluation of Patch Painting Transformer",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--config', default= "./config/train_config_stage2.yaml", help="Model's Config", type=str, dest="config")
parser.add_argument("-m", "--train_mode", default=None, help="Shared File Path For Distributed Learning", dest="train_mode")

args = parser.parse_args()

configFilePath = args.config
train_mode     = args.train_mode

def loadTrainer():
    module = "trainer"
    if train_mode == "ddp":
        exec(f"from {module}.ddp_trainer import Trainer as trainer")
    elif train_mode == "sdxl":
        exec(f"from {module}.sdxl_trainer import Trainer as trainer")
    else:
        exec(f"from {module}.model_trainer import Trainer as trainer")
    
    return eval("trainer")

if __name__ =="__main__":
    trainer = loadTrainer()
    Trainer = trainer(configFilePath)
    Trainer.train()