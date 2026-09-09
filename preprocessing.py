import os
import argparse
import random

from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser(description="Frequency Adaptation Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/train_config.yaml",
        help="Path to the training configuration file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./config/data/",
        help="Directory to save outputs and checkpoints.",
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=0.1,
        help="Test data split rate.",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)

    img_path = config.datasets.img_path
    data_lst = os.listdir(img_path)
    data_len = len(data_lst)
    train_size = int(data_len * (1-args.rate*2))
    rate_size  = int(data_len * args.rate)

    random.shuffle(data_lst)

    with open(os.path.join(args.output_dir, "train.txt"), "w") as f:
        for i in data_lst[:train_size]:
            f.write(i + ",")

    with open(os.path.join(args.output_dir, "valid.txt"), "w") as f:
        for i in data_lst[train_size:train_size+rate_size]:
            f.write(i + ",")

    with open(os.path.join(args.output_dir, "test.txt"), "w") as f:
        for i in data_lst[train_size+rate_size:]:
            f.write(i + ",")
    
    print("="*30)
    print("Data Split Completed!")
    print("="*30)
    print(f"Train Size: {train_size}, Valid Size: {rate_size}, Test Size: {data_len - train_size - rate_size}")