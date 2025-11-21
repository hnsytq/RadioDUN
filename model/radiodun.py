import os
import math

import einops
from tqdm import tqdm
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
import sys
from torch.nn import init

from .mask_generator import RandomMaskingGenerator
from .tools import SeedContextManager
import numpy as np
from .blocks import UnfoldingBlock, ConvBNReLU
from .noise_loss import NoiseLoss
from einops import rearrange


class RadioDUN(nn.Module):

    def __init__(self,
                 img_size=128,
                 sample_number=50,
                 dim=16,
                 num_block=3,
                 para_num=3,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 out_put='./save',
                 ):

        super(RadioDUN, self).__init__()

        self.device = device
        self.sample_num = sample_number
        self.out_put = out_put
        self.num_block = num_block
        self.para_num = para_num
        self.patch_size = img_size
        self.rand_sample = RandomMaskingGenerator(input_size=img_size, sample_number=sample_number)
        for idx in range(self.num_block):
            self.add_module(f'block_{idx}', UnfoldingBlock(dim, self.para_num))

        for idx in range(self.para_num):
            self.add_module(f'init_conv_{idx}', nn.Sequential(
                ConvBNReLU(self.para_num + 1, dim),
                ConvBNReLU(dim, 1)
            ))

        self.PhiT = nn.Parameter(init.xavier_normal_(torch.Tensor(img_size ** 2, sample_number)))
        self.conv_pre = nn.Conv2d(self.para_num, 1, kernel_size=1, bias=False)
        self.shadow_conv = nn.Conv2d(self.para_num-1, 1, kernel_size=1, bias=False)

    def _sample_img(self, img, sample_positions):
        """
        Sample given image at specified positions
        Args:
            img: B x nC x H x W
            sample_positions: B x S x 2 (y, x) samples
        Returns:
            values: B x S x nC
        """
        B, nC, H, W = img.shape
        positions = sample_positions.view(B, 1, -1, 2)
        values = torch.nn.functional.grid_sample(img, positions, align_corners=False)
        values = values.permute(0, 2, 1, 3).contiguous().view(B, -1, nC)
        return values

    def save_model(self, name, out_dir='/content'):
        # First time model is saved (as indicated by not having a pre-existing model directory),
        # create model folder and save model config.

        # Save state dict
        save_model_path = os.path.join(out_dir, f'checkpoints')
        os.makedirs(save_model_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_model_path, f'{name}.pt'))

    def forward(self, inputs):
        # building_mask has 1 for free space, 0 for buildings (may change this for future datasets).

        img, building_map = inputs[0], inputs[2]
        h, w = img.shape[2], img.shape[3]
        sample_map = self.rand_sample(building_map)
        sample_mask = sample_map.to(torch.float32).to(self.device)  # B L N
        img_re = rearrange(img, 'b c h w -> b c (h w)')
        sample = img_re @ sample_mask
        x_init = sample @ sample_mask.transpose(-2, -1)
        x = rearrange(x_init, 'b c (h w) -> b c h w', h=h)
        inputs[-1] = inputs[-1] * torch.max(x)

        paras = []  # building cars noise_map
        xs_maps = []
        for idx in range(self.para_num):
            para_t = torch.cat(inputs[1:], dim=1)
            temp = torch.cat((x, para_t), dim=1)
            paras.append(self.__getattr__(f'init_conv_{idx}')(temp))
            inputs[idx + 1] = paras[-1]

        x = self.conv_pre(torch.cat(paras, dim=1))
        former_info = None
        for idx in range(self.num_block):
            x, paras, former_info = self.__getattr__(f'block_{idx}')(x, paras, sample_mask,
                                                                     self.PhiT, sample, former_info)
            xs_maps.append(x)

        shadowing = self.shadow_conv(torch.cat(paras[1:], dim=1))
        return x, shadowing

    def step(self, batch, optimizer, noise_criterion):
        with torch.set_grad_enabled(True):
            inputs = []
            for key in batch.keys():
                inputs.append(batch[key].to(torch.float32).to(self.device))
            imgs = inputs[0]
            pred_map, noise_map = self.forward(inputs)

            loss_ = nn.functional.mse_loss(pred_map, imgs).to(torch.float32) + noise_criterion(imgs, pred_map,
                                                                                               noise_map)
            loss_.backward()
            optimizer.step()
            optimizer.zero_grad()
        return loss_

    def train_model(self, train_dl, val_dl, optimizer, scheduler, args):

        save_model_dir = self.out_put
        self.to(self.device)

        noise_criterion = NoiseLoss().to(self.device)
        train_losses = []
        psnrs_values = []
        ssims_values = []
        best_loss = np.inf
        epochs = args.num_epochs
        for epoch in range(epochs):
            self.train()
            train_running_loss = 0.0
            pbar = tqdm(train_dl)
            for i, batch in enumerate(pbar):
                pbar.set_description(f'Epoch {epoch + 1}/ {args.num_epochs} - Training:')
                loss = self.step(batch, optimizer, noise_criterion)
                train_running_loss += loss.detach().item()
                # mean loss in a batch
                train_loss = train_running_loss / (i + 1)
                pbar.set_postfix(loss=train_loss)
            with SeedContextManager(seed=args.seed):
                rmse_values, ssim_values, psnr_values = self.eval_model(val_dl)
                if rmse_values < best_loss:
                    best_loss = rmse_values
                    self.save_model('best', out_dir=save_model_dir)

            train_losses.append(train_loss)
            psnrs_values.append(psnr_values)
            ssims_values.append(ssim_values)

            if scheduler:
                scheduler.step()

            self.save_model('last', out_dir=save_model_dir)

        import pickle
        results_dict = {'train_losses': train_losses, 'psnrs_values': psnrs_values, 'ssims_values': ssims_values}
        pickle_path = os.path.join(save_model_dir, f'training_loss')
        os.makedirs(pickle_path, exist_ok=True)
        with open(os.path.join(pickle_path, 'train_losses.pkl'), 'wb') as f:
            pickle.dump(results_dict, f)

    def eval_model(self, test_dl):
        self.eval()
        losses = 0
        pixels = 0

        from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_values = 0
        psnr_values = 0
        total_samples = 0
        with torch.no_grad():
            pbar = tqdm(test_dl)
            self.to(self.device)
            for i, batch in enumerate(pbar):
                pbar.set_description(f'Evaluating batch {i}')
                inputs = []
                for key in batch.keys():
                    inputs.append(batch[key].to(torch.float32).to(self.device))
                imgs = inputs[0]
                pred_map, noise_map = self.forward(inputs)

                pred_map = pred_map.detach()
                loss = nn.functional.mse_loss(pred_map, imgs,
                                              reduction='sum')
                pix = pred_map.numel()
                ssim_value = ssim_metric(pred_map, imgs)
                psnr_value = psnr_metric(pred_map, imgs)

                losses += loss
                pixels += pix

                batch_size = pred_map.size(0)
                ssim_values += ssim_value * batch_size
                psnr_values += psnr_value * batch_size
                total_samples += batch_size

                pbar.set_postfix(loss=math.sqrt(losses / pixels), ssim=ssim_values / total_samples,
                                 psnr=psnr_values / total_samples)
            print(f"SSIM: {ssim_values / total_samples}")
            print(f"PSNR: {psnr_values / total_samples}")
            print(f"RMSE: {math.sqrt(losses / pixels)}")

            return math.sqrt(losses / pixels), ssim_values / total_samples, psnr_values / total_samples
