import os
import numpy as np
from pprint import pprint
from tqdm import tqdm

import torch
import torch.nn as nn

from Models.EHGAMEGAN import Generator, Discriminator, LSTM_AD, GDN  # TAMA structure
from utils.data_loader import get_loader_segment
from utils.optimizer import compute_gradient_penalty
from utils.net_struct import generate_fc_edge_index

# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, model_save_path, patience=3, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.dataset = dataset_name
        self.model_save_path = model_save_path

    def __call__(self, val_loss, model_G, model_D, Predictor, Predictor_S, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_model(val_loss, model_G, model_D, Predictor, Predictor_S, epoch)
        else:
            self.best_score = score
            self.save_model(val_loss, model_G, model_D, Predictor, Predictor_S, epoch)
            self.counter = 0

    def save_model(self, val_loss, model_G, model_D, Predictor, Predictor_S, epoch):
        print('Saving model ...')
        folder = f'{self.model_save_path}_{self.dataset}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'model_G_state_dict': model_G.state_dict(),
            'model_D_state_dict': model_D.state_dict(),
            'model_P_state_dict': Predictor.state_dict(),
            'model_PS_state_dict': Predictor_S.state_dict(),
        }, file_path)
        self.val_loss_min = val_loss


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        """Initialize configurations."""
        self.config = config  # Explicitly store config dictionary
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.device = torch.device(device)
        self.criterion = nn.MSELoss(reduction='none')

        # Generate edge_index_sets for GDN
        fc_edge_index = generate_fc_edge_index(self.input_c)
        fc_edge_index = fc_edge_index.clone().detach().to(self.device, dtype=torch.long)
        self.edge_index_sets = [fc_edge_index]

        _, self.train_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='train',
            dataset=self.dataset
        )
        self.cur_dataset, self.test_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='test',
            dataset=self.dataset
        )

    def train_with_loader(self):
        generator = Generator(win_size=self.win_size, latent_dim=self.latent_dim, input_c=self.input_c).to(self.device)
        discriminator = Discriminator(win_size=self.win_size, input_c=self.input_c).to(self.device)
        predictor_T = LSTM_AD(feats=self.input_c).to(self.device)
        predictor_S = GDN(edge_index_sets=self.edge_index_sets,
                          node_num=self.input_c,
                          win_size=self.win_size,
                          out_layer_num=1,
                          out_layer_inter_dim=self.gat_inter_dim).to(self.device)

        optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        optimizer_PT = torch.optim.Adam(predictor_T.parameters(), lr=self.lr)
        optimizer_PS = torch.optim.Adam(predictor_S.parameters(), lr=self.lr, weight_decay=self.decay)

        if torch.cuda.is_available():
            generator.to(device)
            discriminator.to(device)
            predictor_T.to(device)
            predictor_S.to(device)
            generator.train()
            discriminator.train()
            predictor_T.train()
            predictor_S.train()

        print("======================TRAIN MODE======================")
        path = self.model_save_path
        os.makedirs(path, exist_ok=True)
        early_stopping = EarlyStopping(path, patience=15, verbose=False, dataset_name=self.dataset)
        rec_losses = []
        p_losses = []
        ps_losses = []
        last_mse = 0

        # Check dataloader
        print(f"Total number of batches in train_loader: {len(self.train_loader)}")
        first_batch, _ = next(iter(self.train_loader))
        print(f"First batch shape: {first_batch.shape}")

        for epoch in tqdm(range(self.num_epochs)):
            print(f"\nStarting Epoch {epoch + 1}/{self.num_epochs}")
            for i, (input_data, y_input_data) in enumerate(self.train_loader):
                print(f"  Batch {i + 1}/{len(self.train_loader)}")
                print(f"  Input data shape: {input_data.shape}, y_input_data shape: {y_input_data.shape}")

                optimizer_G.zero_grad()
                optimizer_D.zero_grad()
                optimizer_PT.zero_grad()
                optimizer_PS.zero_grad()
                print("  Gradients zeroed")

                # Shapes:
                # input_data: (B, W, C)
                # y_input_data: (B, H=1, C)
                input_data = input_data.float().to(self.device)
                y_input_data = y_input_data.float().to(self.device)
                print("  Data moved to device")

                if input_data.dim() == 2:
                    input_data = input_data.unsqueeze(-1)  # (B, W, 1)
                if y_input_data.dim() == 1:
                    y_input_data = y_input_data.unsqueeze(-1).unsqueeze(1)  # (B, 1, 1)
                    if self.input_c > 1:
                        y_input_data = y_input_data.expand(-1, 1, self.input_c)

                # Expect: input_data (B, W, C); y_input_data (B, 1, C)
                if input_data.dim() == 2:  # (B, W) -> (B, W, 1)
                    input_data = input_data.unsqueeze(-1)

                if y_input_data.dim() == 1:  # (B,) -> (B, 1, C)
                    y_input_data = y_input_data.unsqueeze(-1)  # (B, 1)
                    y_input_data = y_input_data.unsqueeze(1)  # (B, 1, 1)
                    if self.input_c > 1:
                        y_input_data = y_input_data.expand(-1, 1, self.input_c)

                # LSTM expects (W, B, C)
                input_data_lstm = input_data.transpose(0, 1)  # (W, B, C)
                print(f"  Transposed input for LSTM: {input_data_lstm.shape}")

                # -------------------------
                #  Predictor losses (keep)
                # -------------------------
                p = predictor_T(input_data_lstm)  # p: (W, B, C)
                p_loss = torch.mean(self.criterion(p[-1].unsqueeze(1), y_input_data))  # last step vs horizon=1
                print(f"  Predictor_T loss: {p_loss.item()}")
                p_loss.backward()
                optimizer_PT.step()
                print("  Predictor_T updated")

                ps = predictor_S(input_data, self.edge_index_sets)  # typically (B, 1, C)
                ps_loss = torch.mean(self.criterion(ps, y_input_data))
                print(f"  Predictor_S loss: {ps_loss.item()}")
                ps_loss.backward()
                optimizer_PS.step()
                print("  Predictor_S updated")

                # -------------------------
                #  GAN wiring (window-level)
                # -------------------------
                # Real window as (B, C, W)
                real_input = input_data.permute(0, 2, 1).contiguous()

                # Latent noise for Generator: (B, latent_dim)
                z = torch.randn(input_data.size(0), self.latent_dim, device=self.device)
                print("  Generated z for generator (using real window)")

                # Forward G and D
                fake_input = generator(z)                 # (B, C, W)
                real_validity = discriminator(real_input) # (B, 1)
                fake_validity = discriminator(fake_input) # (B, 1)

                # WGAN-GP discriminator loss
                gradient_penalty = compute_gradient_penalty(discriminator, real_input, fake_input)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
                print(f"  Discriminator loss: {d_loss.item()}")
                d_loss.backward()
                optimizer_D.step()
                print("  Discriminator updated")

                # Generator step
                optimizer_G.zero_grad()
                fake_input = generator(z)
                fake_validity = discriminator(fake_input)
                g_loss = -torch.mean(fake_validity)
                print(f"  Generator loss: {g_loss.item()}")
                g_loss.backward()
                optimizer_G.step()
                print("  Generator updated")

                # Reconstruction monitor for logging (window-level)
                rec_loss = torch.mean(self.criterion(fake_input, real_input))
                rec_losses.append(rec_loss.detach().cpu().numpy())
                print(f"  Reconstruction loss (window): {rec_loss.item()}")

                p_losses.append(p_loss.detach().cpu().numpy())
                ps_losses.append(ps_loss.detach().cpu().numpy())

            if epoch % 1 == 0:
                mse = np.average(rec_losses)
                tqdm.write(
                    "Epoch: {0}, Steps: {1} | g_loss: {2:.7f} d_loss: {3:.7f} MSE: {4:.7f} SPD: {5:.7f} P_MSE: {6:.7f} PS_MSE: {7:.7f}".format(
                        epoch + 1, i, g_loss.item(), d_loss.item(), mse, last_mse - mse, np.average(p_losses), np.average(ps_losses)))
                last_mse = mse

            early_stopping(mse, generator, discriminator, predictor_T, predictor_S, epoch)
            if early_stopping.early_stop:
                print("Early stopping with patience ", early_stopping.patience)
                break

        # Save checkpoint
        print("Saving checkpoint...")
        torch.save({
            'model_G_state_dict': generator.state_dict(),
            'model_D_state_dict': discriminator.state_dict(),
            'model_L_state_dict': predictor_T.state_dict(),
            'model_GDN_state_dict': predictor_S.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'optimizer_PT_state_dict': optimizer_PT.state_dict(),
            'optimizer_PS_state_dict': optimizer_PS.state_dict(),
        }, os.path.join(path, 'model.ckpt'))
        print("Checkpoint saved.")

    def test(self):
        """Test the EH-GAM-EGAN model (not used in TAMA preprocessing)."""
        self.G = Generator(win_size=self.win_size, latent_dim=self.latent_dim, input_c=self.input_c).to(self.device)
        self.D = Discriminator(win_size=self.win_size, input_c=self.input_c).to(self.device)
        self.L = LSTM_AD(feats=self.input_c).to(self.device)
        self.GDN = GDN(edge_index_sets=self.edge_index_sets, node_num=self.input_c, win_size=self.win_size,
                       out_layer_num=1, out_layer_inter_dim=self.gat_inter_dim).to(self.device)

        fname = f'{self.model_save_path}_{self.dataset}/model.ckpt'
        checkpoint = torch.load(fname, map_location=self.device)

        self.G.load_state_dict(checkpoint['model_G_state_dict'])
        self.D.load_state_dict(checkpoint['model_D_state_dict'])
        self.L.load_state_dict(checkpoint['model_P_state_dict'])
        self.GDN.load_state_dict(checkpoint['model_PS_state_dict'])

        self.G.eval()
        self.D.eval()
        self.L.eval()
        self.GDN.eval()

        print("======================TEST MODE======================")
        criterion = nn.MSELoss(reduction='none')

        test_labels = []
        test_energy = []
        p_energy = []
        ps_energy = []
        g_energy = []
        d_energy = []

        for i, (input_data, y_input_data) in enumerate(self.test_loader):
            # input_data: (B, W, C), y_input_data: (B, 1, C)
            input_data = input_data.float().to(self.device)
            y_input_data = y_input_data.float().to(self.device)

            # ----- Generator/Discriminator on full window -----
            real_input = input_data.permute(0, 2, 1).contiguous()  # (B, C, W)
            z = torch.randn(input_data.size(0), self.latent_dim, device=self.device)
            fake_input = self.G(z)

            # Window reconstruction error -> per-channel scalar, then reshape to (B,1,C)
            g_loss = torch.mean(criterion(real_input, fake_input), dim=2).unsqueeze(1)  # (B, 1, C)

            # Discriminator score -> convert to positive loss-like signal, reshape to (B,1,C)
            d_score = self.D(real_input)  # (B, 1)
            d_loss = (-d_score + 1.0).unsqueeze(-1).expand(-1, 1, self.input_c)  # (B, 1, C)

            # ----- Predictors (keep shapes consistent with training) -----
            input_data_lstm = input_data.transpose(0, 1)  # (W, B, C)
            p = self.L(input_data_lstm)                   # (W, B, C)
            p_loss = criterion(p[-1].unsqueeze(1), y_input_data)  # (B, 1, C)

            ps = self.GDN(input_data, self.edge_index_sets)        # (B, 1, C)
            ps_loss = criterion(ps, y_input_data)                  # (B, 1, C)

            # ----- Combine (per original weighting) -----
            loss = self.alpha * ps_loss + (1 - self.alpha) * p_loss + self.beta * g_loss + (1 - self.beta) * d_loss
            loss = torch.mean(loss, dim=1)                 # (B, C)

            # Reduce per-component for logging
            p_loss = torch.mean(p_loss, dim=1)             # (B, C)
            p_loss = torch.mean(p_loss, dim=-1)            # (B,)
            ps_loss = torch.mean(ps_loss, dim=1)           # (B, C)
            ps_loss = torch.mean(ps_loss, dim=-1)          # (B,)
            d_loss = torch.mean(torch.mean(d_loss, dim=1), dim=-1)  # (B,)
            g_loss = torch.mean(torch.mean(g_loss, dim=1), dim=-1)  # (B,)

            win_loss = torch.mean(loss, dim=-1)            # (B,)

            test_energy.append(win_loss.detach().cpu().numpy())
            p_energy.append(p_loss.detach().cpu().numpy())
            ps_energy.append(ps_loss.detach().cpu().numpy())
            d_energy.append(d_loss.detach().cpu().numpy())
            g_energy.append(g_loss.detach().cpu().numpy())

        test_energy = np.concatenate(test_energy, axis=0).reshape(-1)
        p_energy = np.concatenate(p_energy, axis=0).reshape(-1)
        ps_energy = np.concatenate(ps_energy, axis=0).reshape(-1)
        g_energy = np.concatenate(g_energy, axis=0).reshape(-1)
        d_energy = np.concatenate(d_energy, axis=0).reshape(-1)

        test_labels = self.cur_dataset.test_labels[self.win_size:]
        print("test_labels:     ", test_labels.shape)

        np.save('./energy/test_energy_' + self.dataset, test_energy)
        np.save('./labels/test_labels_' + self.dataset, test_labels)
        np.save('./energy/p_energy_' + self.dataset, p_energy)
        np.save('./energy/ps_energy_' + self.dataset, ps_energy)
        np.save('./energy/g_energy_' + self.dataset, g_energy)
        np.save('./energy/d_energy_' + self.dataset, d_energy)

        score = test_energy
        label = test_labels
        print('aa', score.shape, label.shape)
        # bf_eval = bf_search(score, label, start=0.001, end=1, step_num=150, verbose=False)
        # pprint(bf_eval)