import os
import argparse
import random
import numpy as np
import torch
from utils import SeedContextManager
from model.radiodun import RadioDUN
from data.mix_dataloader import setting_dataset


def get_args():
    parser = argparse.ArgumentParser(description='Parameters for RadioDUN')
    parser.add_argument('--data_dir', type=str, default='./data/RadioMapSeer_Indices_mix.pkl')
    parser.add_argument('--output_dir', type=str, default='./radiodun')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epoch to train')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--sample_num', type=int, default=9, help='sample number')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--model_phase', type=str, default='train', help='train or test')
    parser.add_argument('--img_size', type=int, default=256, help='image size for dataset')
    parser.add_argument('--model_path', type=str, default=r'./checkpoints/best.pt', help='model path')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--gain_dirs', type=str, default='carsDPM', help='simulation types')
    parser.add_argument('--num_block', type=int, default=3, help='the number of unfolding blocks')
    parser.add_argument('--para_num', type=int, default=3, help='the number of environmental factors')
    parser.add_argument('--dim', type=int, default=16, help='the number of feature channels')
    parser.add_argument('--val_antennas', type=bool, default=False, help='transmitter position available')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_dataset, val_dataset, test_dataset = setting_dataset(args)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.num_workers)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)

    model = RadioDUN(
        img_size=args.img_size,
        sample_number=args.sample_num,
        dim=args.dim,
        num_block=args.num_block,
        para_num=args.para_num,
        device=args.device,
        out_put=args.output_dir,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.num_epochs)

    if args.model_phase == 'train':
        print(
            '----------------------------------------------Start training---------------------------------------------')
        model.train_model(train_dl, val_dl, optim, scheduler, args=args)
        model.load_state_dict(torch.load(args.output_dir + '/checkpoints/best.pt'))
        metrics_str = ''
        rmse, ssim, psnr = model.eval_model(val_dl)
        metrics_str += 'val: \nRMSE: %.4f, SSIM: %.4f, PSNR: %.4f\n' % (rmse, ssim, psnr)
        rmse, ssim, psnr = model.eval_model(test_dl)
        metrics_str += 'test: \nRMSE: %.4f, SSIM: %.4f, PSNR: %.4f\n' % (rmse, ssim, psnr)
        with open(os.path.join(args.output_dir, 'metrics.txt'), 'w+') as f:
            f.write(metrics_str)
            f.close()
        print(
            '----------------------------------------------Finished training------------------------------------------')
    elif args.model_phase == 'test':
        model.load_state_dict(torch.load(args.model_path))
        with SeedContextManager(args.seed):
            print(
                '----------------------------------------------Start testing------------------------------------------')
            rmse, ssim, psnr = model.eval_model(test_dl)
            print('RMSE: %.4f, SSIM: %.4f, PSNR: %.4f' % (rmse, ssim, psnr))
            print('----------------------------------------------Finished testing----------')
