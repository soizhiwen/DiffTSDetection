import argparse
import torch
from training_utils import MyDataset, fetchModel, fetchDiffusionConfig
import numpy as np
from torch import from_numpy, device
from torch.utils.data import DataLoader
import os
from data_utils import Preprocessor
import random


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', type=str,
                        help='MetroTraffic, BeijingAirQuality, AustraliaTourism, RossmanSales, PanamaEnergy',
                        required=True)
    parser.add_argument('-proportion', type=float, default=0.6, help='proportion of training data')
    parser.add_argument('-backbone', type=str, help='Transformer, Bilinear, Linear, S4', default='wavestitch')
    parser.add_argument('-beta_0', type=float, default=0.0001, help='initial variance schedule')
    parser.add_argument('-beta_T', type=float, default=0.02, help='last variance schedule')
    parser.add_argument('-timesteps', '-T', type=int, default=200, help='training/inference timesteps')
    parser.add_argument('-hdim', type=int, default=64, help='hidden embedding dimension')
    parser.add_argument('-batch_size', type=int, help='batch size', default=512)
    parser.add_argument('-layers', type=int, default=4, help='number of hidden layers')
    parser.add_argument('-window_size', type=int, default=32, help='the size of the training windows')
    # parser.add_argument('-stride', type=int, default=1, help='the stride length to shift the training window by')
    parser.add_argument('-num_res_layers', type=int, default=4, help='the number of residual layers')
    parser.add_argument('-res_channels', type=int, default=64, help='the number of res channels')
    parser.add_argument('-skip_channels', type=int, default=64, help='the number of skip channels')
    parser.add_argument('-diff_step_embed_in', type=int, default=32, help='input embedding size diffusion')
    parser.add_argument('-diff_step_embed_mid', type=int, default=64, help='middle embedding size diffusion')
    parser.add_argument('-diff_step_embed_out', type=int, default=64, help='output embedding size diffusion')
    parser.add_argument('-s4_lmax', type=int, default=100)
    parser.add_argument('-s4_dstate', type=int, default=64)
    parser.add_argument('-s4_dropout', type=float, default=0.0)
    parser.add_argument('-s4_bidirectional', type=bool, default=True)
    parser.add_argument('-s4_layernorm', type=bool, default=True)
    args = parser.parse_args()
    dataset = args.dataset
    device = device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = Preprocessor(dataset, args.window_size, proportion=args.proportion)
    d_vals_tensor = from_numpy(preprocessor.train_data)
    training_samples = d_vals_tensor.unfold(0, args.window_size, 1).transpose(1, 2)
    test_d_vals_tensor = from_numpy(preprocessor.test_data)
    test_samples = test_d_vals_tensor.unfold(0, args.window_size, 1).transpose(1, 2)
    total_samples = len(training_samples) + len(test_samples)
    in_dim = preprocessor.var_num
    out_dim = preprocessor.var_num
    model = fetchModel(in_dim, out_dim, args).to(device)
    diffusion_config = fetchDiffusionConfig(args)

    saved_params = torch.load(f'checkpoints/{args.dataset}_{args.window_size}/wavestitch_checkpoint.pth', map_location=device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(saved_params[name])
            param.requires_grad = False

    model.eval()
    samples = np.empty([0, args.window_size, preprocessor.var_num])
    B, L, C = args.batch_size, args.window_size, preprocessor.var_num
    with torch.no_grad():
        while len(samples) < total_samples:
            x = torch.normal(0, 1, (B, L, C)).to(device)
            for step in range(diffusion_config['T'] - 1, -1, -1):
                times = torch.full(size=(x.shape[0], 1), fill_value=step).to(device)
                alpha_bar_t = diffusion_config['alpha_bars'][step].to(device)
                alpha_bar_t_1 = diffusion_config['alpha_bars'][step - 1].to(device)
                alpha_t = diffusion_config['alphas'][step].to(device)
                beta_t = diffusion_config['betas'][step].to(device)
                epsilon_pred = model(x, times)
                epsilon_pred = epsilon_pred.permute((0, 2, 1))
                if step > 0:
                    vari = beta_t * ((1 - alpha_bar_t_1) / (1 - alpha_bar_t)) * torch.normal(0, 1, size=epsilon_pred.shape).to(device)
                else:
                    vari = 0.0

                normal_denoising = (x - ((beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred)) / torch.sqrt(alpha_t)
                normal_denoising += vari
                x = normal_denoising

            samples = np.vstack([samples, x.detach().cpu().numpy()])
            samples = samples[:total_samples]

    path = f'datasets/wavestitch/'
    if not os.path.exists(path):
        os.makedirs(path)

    np.savez_compressed(
        os.path.join(path, f"{args.dataset}_{args.window_size}"),
        norm_synth=samples,
    )
