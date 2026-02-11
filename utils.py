import os
import random

import cv2
import numpy as np
import torch


class SeedContextManager:
    def __init__(self, seed=None):
        self.seed = seed + 40 if seed is not None else None
        self.random_state = None
        self.np_random_state = None
        self.torch_random_state = None
        self.torch_cuda_random_state = None

    def __enter__(self):
        if self.seed is not None:
            self.random_state = random.getstate()
            self.np_random_state = np.random.get_state()
            self.torch_random_state = torch.random.get_rng_state()
            if torch.cuda.is_available():
                self.torch_cuda_random_state = torch.cuda.random.get_rng_state_all()

            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seed is not None:
            random.setstate(self.random_state)
            np.random.set_state(self.np_random_state)
            torch.random.set_rng_state(self.torch_random_state)
            if torch.cuda.is_available():
                torch.cuda.random.set_rng_state_all(self.torch_cuda_random_state)


def save_model(self, name, out_dir='/content'):
    # First time model is saved (as indicated by not having a pre-existing model directory),
    # create model folder and save model config.

    # Save state dict
    save_model_path = os.path.join(out_dir, f'checkpoints')
    os.makedirs(save_model_path, exist_ok=True)
    torch.save(self.state_dict(), os.path.join(save_model_path, f'{name}.pt'))


def main_files_sel():
    gain_folder = 'D:/PP_Data/PP5D_latest/PPData5D-success/png/03.Ocean/'
    bui_folder = 'D:/PP_Data/PP5D_latest/PPData5D-success/npz/'

    gain_files = os.listdir(gain_folder)
    bui_files = os.listdir(bui_folder)
    save_bui_files, save_gain_files = [], []
    for name in bui_files:
        if 'T03' not in name:
            continue
        t_name = name[:-8]
        # print(t_name)
        t_name = t_name + 'f00_ss_z00.png'
        if t_name in gain_files:
            save_gain_files.append(t_name)
            save_bui_files.append(name)
        else:
            print(f'Error! {name}')
            continue
    print(len(save_bui_files), len(save_gain_files))
    from sklearn.model_selection import train_test_split
    bui_train, bui_val, gain_train, gain_val = train_test_split(save_bui_files, save_gain_files, test_size=0.2)
    bui_no_sup, bui_sup, gain_no_sup, gain_sup = train_test_split(bui_train, gain_train, test_size=0.292)
    print(len(bui_sup), len(gain_sup), len(gain_no_sup), len(bui_no_sup), len(bui_val), len(gain_val))
    files_dict = dict(bui_sup=bui_sup, gain_sup=gain_sup, bui_no_sup=bui_no_sup, gain_no_sup=gain_no_sup,
                      bui_val=bui_val, gain_val=gain_val)

    import json
    with open('../data/PP5D_ocean.json', 'w') as f:
        json.dump(files_dict, f)


if __name__ == '__main__':
    # from PIL import Image
    # gain_files = 'D:/PP_Data/PP5D_latest/PPData5D-success/png/03.Ocean/T03C0D0000_n00_f00_ss_z01.png'
    #
    # gain_map = Image.open(gain_files)
    #
    # print(gain_map.size, np.max(gain_map))
    # gain_map = np.asarray(gain_map) / 255
    # print(np.max(gain_map))
    main_files_sel()
