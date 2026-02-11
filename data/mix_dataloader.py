import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import random
import pickle
import numpy as np

base_folder = '/datasets/'


class MixDataset(Dataset):
    def __init__(self, gain_dirs, building_dir=base_folder + 'RadioMapSeer/png/buildings_complete',
                 root_dir=base_folder + 'RadioMapSeer/gain',
                 cars_dir=base_folder + 'RadioMapSeer/png/cars',
                 antennas_dir=base_folder + 'RadioMapSeer/png/antennas',
                 img_size=256, transform=None, val_ant=True):
        self.root_dir = root_dir
        self.gain_dirs = gain_dirs
        self.building_dir = building_dir
        self.cars_dir = cars_dir
        self.antennas_dir = antennas_dir
        self.val_ant = val_ant
        self.h, self.w = np.indices((img_size, img_size))
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()])
        else:
            self.transform = transform
        self._imgs_list = []
        for gain_dir in gain_dirs:
            for img in os.listdir(os.path.join(self.root_dir, gain_dir)):
                self._imgs_list.append(os.path.join(self.root_dir, gain_dir, img))

    def __len__(self):
        return len(self._imgs_list)

    def __getitem__(self, index):
        img_path = self._imgs_list[index]
        img_name = os.path.basename(img_path)
        building_img_name = img_name.split('_')[0] + '.png'
        building_img_path = os.path.join(self.building_dir, building_img_name)
        cars_img_path = os.path.join(self.cars_dir, building_img_name)
        antennas_img_path = os.path.join(self.antennas_dir, img_name)

        if self.val_ant:
            with open(antennas_img_path, 'rb') as a:
                antennas_img = Image.open(a)
                index_h, index_w = np.nonzero(antennas_img)
        else:
            index_h, index_w = 0, 0
        abs_pos = np.log10(np.sqrt((self.h - index_h) ** 2 + (self.w - index_w) ** 2) + 1)
        abs_pos = 1 - (abs_pos - np.min(abs_pos)) / (np.max(abs_pos) - np.min(abs_pos))
        abs_pos = self.transform(abs_pos)

        with open(cars_img_path, 'rb') as c:
            ori_cars_img = Image.open(c)
            cars_img = self.transform(ori_cars_img)

        with open(building_img_path, 'rb') as b:
            ori_building_img = Image.open(b)
            success = False
            while not success:
                if self.transform is not None:
                    seed = torch.random.seed()
                    torch.random.manual_seed(seed)
                    building_img = self.transform(ori_building_img)
                    if (1 - building_img).sum() > 1000:
                        success = True
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            if self.transform is not None:
                torch.random.manual_seed(seed)
                img = self.transform(img)

        elem = {'img': img, 'los': abs_pos, 'building_img': building_img, 'car_img': cars_img}
        return elem


def setting_dataset(args, test_size=0.2, val_size=0.05):
    dataset = MixDataset(gain_dirs=[args.gain_dirs], val_ant=args.val_antennas)
    data_pkl = args.data_dir
    try:
        with open(os.path.join(data_pkl), 'rb') as f:
            indices = pickle.load(f)
            train_imgs, val_imgs, test_imgs = indices[0], indices[1], indices[2]
    except:
        all_imgs = dataset._imgs_list

        all_building_ids = [i for i in range(0, 700)]
        test_building_number = random.sample(all_building_ids, int(700 * test_size))

        test_imgs = [img for img in all_imgs if int(os.path.basename(img).split('_')[0]) in test_building_number]

        train_val_imgs = [img for img in all_imgs if
                          not int(os.path.basename(img).split('_')[0]) in test_building_number]

        train_size = int(((1 - test_size - val_size) / (1 - test_size)) * len(train_val_imgs))
        val_size_actual = len(train_val_imgs) - train_size

        random.shuffle(train_val_imgs)

        train_imgs = train_val_imgs[:train_size]
        val_imgs = train_val_imgs[train_size:train_size + val_size_actual]

        indices = [train_imgs, val_imgs, test_imgs]
        with open(os.path.join(data_pkl), 'wb') as f:
            pickle.dump(indices, f)

    class SubsetDataset(Dataset):
        def __init__(self, imgs_list):
            self.imgs_list = imgs_list

        def __len__(self):
            return len(self.imgs_list)

        def __getitem__(self, index):
            return dataset.__getitem__(index)

    train_dataset = SubsetDataset(train_imgs)
    val_dataset = SubsetDataset(val_imgs)
    test_dataset = SubsetDataset(test_imgs)

    return train_dataset, val_dataset, test_dataset
