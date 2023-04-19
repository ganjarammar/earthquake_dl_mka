import os
import sys
import json
import argparse

sys.path.append('../')

from IndoEQ.trainer import trainer
from IndoEQ.config import TrainingConfig

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config_path', type=str, required=True)
    args = args.parse_args()
    config = json.load(open(args.config_path))
    config = TrainingConfig(**config)
    print(json.dumps(config.__dict__, indent=4))

    trainer(**config.__dict__)
