# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import random
import torch


class RandomMaskingGenerator:
    def __init__(self, input_size, sample_number):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_sample = sample_number

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_sample
        )
        return repr_str

    def normalize_coords(self, coords, H, W):
        xs, ys = torch.split(coords, 1, dim=-1)
        delta_y = 1. / H
        delta_x = 1. / W
        xs = (-1 + delta_x) + xs * 2.0 / W
        ys = (-1 + delta_y) + ys * 2.0 / H

        return torch.cat([xs, ys], dim=-1)

    def __call__(self, building):
        '''
        building_img: (b,h,w) sample point from 0
        mask: (b,h*w)
        '''
        building = building.squeeze(1)
        B, H, W = building.shape
        sample = torch.zeros((B, H * W, self.num_sample), dtype=torch.uint8)
        start = 0
        building = building.view(B, -1)
        zero_inds = torch.nonzero(building == 0, as_tuple=True)
        for i in range(B):
            end = (zero_inds[0] == i).sum().item()
            sample_inds = torch.tensor(random.sample(range(start, start + end - 1), self.num_sample))
            for idx in range(sample_inds.shape[0]):
                sample[zero_inds[0][sample_inds[idx]], zero_inds[1][sample_inds[idx]], idx] = 1
            start += end

        return sample
