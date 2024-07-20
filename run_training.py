import logging.config
from pathlib import Path
import torch
import torch.multiprocessing as mp
import random

from dp.preprocess import preprocess
from dp.train import train
from dp.utils.io import read_config


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().strip('(),')
            parts = line.split(', ')
            if len(parts) == 3:
                lang, word, pronunciation = parts
                data.append((lang.strip("'"), word.strip("'"), pronunciation.strip("'")))
    return data


def split_data(data, train_ratio=0.8):
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]


if __name__ == '__main__':
    config_file_path = Path('logging.yaml')
    config = read_config(config_file_path)
    logging.config.dictConfig(config)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # データを読み込む
    all_data = load_data('cmudict-0.7b-ipa_convert.txt')

    # データを分割する
    train_data, val_data = split_data(all_data)

    config_file = 'dp/configs/forward_config.yaml'

    preprocess(config_file=config_file,
               train_data=train_data,
               val_data=val_data,
               deduplicate_train_data=False)

    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(num_gpus, config_file))
    else:
        train(rank=0, num_gpus=num_gpus, config_file=config_file)
