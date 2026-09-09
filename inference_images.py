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
parser.add_argument("-p","--unet_path", default="/workspace/logs/Frequency_Adaptation/ckpt_dir/SAModel/", help="Unet's Checkpoint Path For Inference", type=str, dest="unet_path")
parser.add_argument("-cn", "--checkpoint_num", default=37000, help="Checkpoint Number to Load for Inference", type=int, dest="checkpoint_num")
parser.add_argument("-f", "--filter_scale", default=[0.0], help="Frequency Filter Scale for Inference", type=str, dest="filter_scale")
parser.add_argument("-n", "--num_samples", default=5, help="Number of Samples to Generate", type=int, dest="num_samples")
parser.add_argument("-s", "--cfg_scale", default=7.5, help="Classifier Free Guidance Scale for Inference", type=float, dest="cfg_scale")
parser.add_argument("-i", "--indexes", default=[0], help="Indexes of Samples to Generate (Comma Separated)", type=str, dest="indexes")
parser.add_argument("-o", "--output_path", default="./output", help="Output Path for Generated Images", type=str, dest="output_path")   
args = parser.parse_args()

configFilePath = args.config
train_mode     = args.train_mode
unet_path      = os.path.join(args.unet_path, f"checkpoint-{args.checkpoint_num}", "unet", "diffusion_pytorch_model.safetensors")

def loadTrainer():
    module = "trainer"
    if train_mode == "ddp":
        exec(f"from {module}.ddp_trainer import Trainer as trainer")
    else:
        exec(f"from {module}.model_trainer import Trainer as trainer")
    
    return eval("trainer")

if __name__ =="__main__":
    trainer = loadTrainer()
    Trainer = trainer(configFilePath)
    Trainer.inference(
        unet_path = unet_path,
        filter_scale = [float(x) for x in args.filter_scale.split(",")] if args.indexes else [],
        num_samples=args.num_samples, 
        cfg_scale=args.cfg_scale,
        indexes = [int(x) for x in args.indexes.split(",")] if args.indexes else [],
        output_path = args.output_path)