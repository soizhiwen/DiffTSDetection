import argparse

import torch
import random

from data_utils import Preprocessor
from training_utils import MyDataset, fetchModel, fetchDiffusionConfig
import numpy as np

from torch import from_numpy, optim, nn, randint, normal, sqrt, device
import os
from torch.utils.data import DataLoader

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
                        help='MetroTraffic, BeijingAirQuality, AustraliaTourism, RossmanSales, PanamaEnergy', required=True)
    parser.add_argument('-proportion', type=float, default=0.6, help='proportion of training data')
    parser.add_argument('-backbone', type=str, help='Transformer, Bilinear, Linear, S4', default='wavestitch')
    parser.add_argument('-beta_0', type=float, default=0.0001, help='initial variance schedule')
    parser.add_argument('-beta_T', type=float, default=0.02, help='last variance schedule')
    parser.add_argument('-timesteps', '-T', type=int, default=200, help='training/inference timesteps')
    parser.add_argument('-hdim', type=int, default=64, help='hidden embedding dimension')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-batch_size', type=int, help='batch size', default=64)
    parser.add_argument('-epochs', type=int, default=1000, help='training epochs')
    parser.add_argument('-layers', type=int, default=4, help='number of hidden layers')
    parser.add_argument('-window_size', type=int, default=32, help='the size of the training windows')
    parser.add_argument('-stride', type=int, default=1, help='the stride length to shift the training window by')
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
    in_dim = preprocessor.var_num
    out_dim = preprocessor.var_num
    training_dataset = MyDataset(training_samples.float())
    model = fetchModel(in_dim, out_dim, args).to(device)
    diffusion_config = fetchDiffusionConfig(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    """TRAINING"""
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            timesteps = randint(diffusion_config['T'], size=(batch.shape[0],)).to(device)
            sigmas = normal(0, 1, size=batch.shape).to(device)
            """Forward noising"""
            alpha_bars = diffusion_config['alpha_bars'].to(device)
            coeff_1 = sqrt(alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))
            coeff_2 = sqrt(1 - alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))
            batch_noised = coeff_1 * batch + coeff_2 * sigmas
            batch_noised = batch_noised.to(device)
            timesteps = timesteps.reshape((-1, 1))
            sigmas_predicted = model(batch_noised, timesteps).permute((0, 2, 1))
            optimizer.zero_grad()
            loss = criterion(sigmas_predicted, sigmas)
            loss.backward()
            total_loss += loss
            optimizer.step()
        print(f'epoch: {epoch}, loss: {total_loss}')
    path = f'checkpoints/{args.dataset}_{args.window_size}/'
    filename = "wavestitch_checkpoint.pth"
    filepath = os.path.join(path, filename)

    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), filepath)
