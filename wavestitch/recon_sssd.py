import argparse
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
    parser.add_argument('-variant', type=str, help='real, diffusionts, etc.', required=True)
    parser.add_argument('-mode', type=int, help='train set (0) or test set (1).', required=True)
    parser.add_argument('-proportion', type=float, default=0.6, help='proportion of training data')
    parser.add_argument('-backbone', type=str, help='Transformer, Bilinear, Linear, S4', default='sssd')
    parser.add_argument('-beta_0', type=float, default=0.0001, help='initial variance schedule')
    parser.add_argument('-beta_T', type=float, default=0.02, help='last variance schedule')
    parser.add_argument('-timesteps', '-T', type=int, default=200, help='training/inference timesteps')
    parser.add_argument('-num_inference_steps', type=int, default=50, help='DDIM inference timesteps')
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
    if args.mode == 0:
        d_vals_tensor = from_numpy(preprocessor.train_data)
        training_samples = d_vals_tensor.unfold(0, args.window_size, 1).transpose(1, 2)
        if args.variant != "real":
            training_samples = np.load(f"./datasets/{args.variant}/{dataset}_{args.window_size}.npz")["norm_synth"][:len(training_samples)]
            training_samples = from_numpy(training_samples)
    elif args.mode == 1:
        d_vals_tensor = from_numpy(preprocessor.test_data)
        training_samples = d_vals_tensor.unfold(0, args.window_size, 1).transpose(1, 2)
        if args.variant != "real":
            training_samples = np.load(f"./datasets/{args.variant}/{dataset}_{args.window_size}.npz")["norm_synth"][-len(training_samples):]
            training_samples = from_numpy(training_samples)
    else:
        raise ValueError("mode should be either 0 (train set) or 1 (test set).")
    in_dim = preprocessor.var_num
    out_dim = preprocessor.var_num
    train_dataset = MyDataset(training_samples.float())
    model = fetchModel(in_dim, out_dim, args).to(device)
    diffusion_config = fetchDiffusionConfig(args)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    saved_params = torch.load(f'checkpoints/{args.dataset}_{args.window_size}/sssd_checkpoint.pth', map_location=device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(saved_params[name])
            param.requires_grad = False

    model.eval()
    total_timesteps = diffusion_config['T']
    num_inference_steps = args.num_inference_steps
    timesteps = torch.linspace(0, total_timesteps, num_inference_steps+1, dtype=torch.long)[:-1] + 1
    reversed_timesteps = timesteps.flip(0) # large to small
    original = np.empty([0, args.window_size, preprocessor.var_num])
    samples = np.empty([0, args.window_size, preprocessor.var_num])
    with torch.no_grad():
        for test_batch in dataloader:
            original = np.vstack([original, test_batch.detach().cpu().numpy()])
            test_batch = test_batch.to(device)
            conditional_mask = torch.zeros_like(test_batch).to(device)
            conditional_signal = torch.zeros_like(test_batch).to(device)

            # DDIM Inversion
            for i in range(1, num_inference_steps):
                t = timesteps[i]
                time_cond = torch.full((test_batch.shape[0], 1), t, device=device, dtype=torch.long)
                noise_pred = model((test_batch, conditional_signal, conditional_mask, time_cond)).permute((0, 2, 1))

                current_t = max(0, t.item() - (total_timesteps//num_inference_steps)) #t
                next_t = t # min(999, t.item() + (1000//num_inference_steps)) # t+1
                alpha_t = diffusion_config["alpha_bars"][current_t].to(device)
                alpha_t_next = diffusion_config["alpha_bars"][next_t].to(device)
                test_batch = (test_batch - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred

            # DDIM Generation
            for i in range(0, num_inference_steps):
                t = reversed_timesteps[i]
                time_cond = torch.full((test_batch.shape[0],1), t, device=device, dtype=torch.long)
                noise_pred = model((test_batch, conditional_signal, conditional_mask, time_cond)).permute((0, 2, 1))

                prev_t = max(1, t.item() - (total_timesteps//num_inference_steps)) # t-1
                alpha_t = diffusion_config["alpha_bars"][t.item()].to(device)
                alpha_t_prev = diffusion_config["alpha_bars"][prev_t].to(device)
                predicted_x0 = (test_batch - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
                direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
                test_batch = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt

            samples = np.vstack([samples, test_batch.detach().cpu().numpy()])


    path = f'outputs/recon_{args.dataset}_{args.variant}_sssd_{args.window_size}_{args.mode}/'
    if not os.path.exists(path):
        os.makedirs(path)

    losses = ((original - samples) ** 2).reshape(original.shape[0], -1)
    df = pd.DataFrame(losses).astype(np.float32)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, os.path.join(path, f"recon_{args.dataset}_{args.variant}_sssd_{args.window_size}_{args.mode}.parquet"), compression="zstd")

    # np.savez_compressed(
    #     os.path.join(path, f"recon_{args.dataset}_{args.variant}_sssd_{args.window_size}_{args.mode}"),
    #     norm_orig=original,
    #     norm_recon=samples,
    # )
