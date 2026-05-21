import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_sine_cosine(v, num_terms):
    num_terms = torch.tensor(num_terms).to(device)
    v = v.to(device)

    # Compute the angles for all terms
    angles = (
        2 ** torch.arange(num_terms).float().to(device)
        * torch.tensor(math.pi).to(device)
        * v.unsqueeze(-1)
    )

    # Compute sine and cosine values for all angles
    sine_values = torch.sin(angles)
    cosine_values = torch.cos(angles)

    # Reshape sine and cosine values for concatenation
    sine_values = sine_values.reshape(*sine_values.shape[:-2], -1)
    cosine_values = cosine_values.reshape(*cosine_values.shape[:-2], -1)

    # Concatenate sine and cosine values along the last dimension
    result = torch.cat((sine_values, cosine_values), dim=-1)

    return result


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.RNN = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, d_last_states = self.RNN(x)
        y_hat_logit = self.fc(d_last_states[-1])
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat


class Embedding_data(nn.Module):
    def __init__(self, input_size, emb_dim, n_bins, n_cats, n_nums, cards):
        super().__init__()

        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.cards = cards

        self.n_disc = self.n_bins + self.n_cats
        self.num_categorical_list = [2] * self.n_bins + self.cards

        if self.n_disc != 0:
            # Create a list to store individual embeddings
            self.embeddings_list = nn.ModuleList()

            # Create individual embeddings for each variable
            for num_categories in self.num_categorical_list:
                embedding = nn.Embedding(num_categories, emb_dim)
                self.embeddings_list.append(embedding)

        if self.n_nums != 0:
            self.mlp_nums = nn.Sequential(
                nn.Linear(
                    16 * n_nums, 16 * n_nums
                ),  # this should be 16 * n_nums, 16 * n_nums
                nn.SiLU(),
                nn.Linear(16 * n_nums, 16 * n_nums),
            )

        self.mlp_output = nn.Sequential(
            nn.Linear(
                emb_dim * self.n_disc + 16 * n_nums, emb_dim
            ),  # this should be 16 * n_nums, 16 * n_nums
            nn.ReLU(),
            nn.Linear(emb_dim, input_size),
        )

    def forward(self, x):
        x_disc = x[:, :, 0 : self.n_disc].long().to(device)
        x_nums = x[:, :, self.n_disc : self.n_disc + self.n_nums].to(device)

        x_emb = torch.Tensor().to(device)

        # Binary + Discrete Variables
        if self.n_disc != 0:
            variable_embeddings = [
                embedding(x_disc[:, :, i])
                for i, embedding in enumerate(self.embeddings_list)
            ]
            x_disc_emb = torch.cat(variable_embeddings, dim=2)
            x_emb = x_disc_emb

        # Numerical Variables
        if self.n_nums != 0:
            x_nums = compute_sine_cosine(x_nums, num_terms=8)
            x_nums_emb = self.mlp_nums(x_nums)
            x_emb = torch.cat([x_emb, x_nums_emb], dim=2)

        final_emb = self.mlp_output(x_emb)

        return final_emb


class DeapStack(nn.Module):
    def __init__(
        self,
        channels,
        n_bins,
        n_cats,
        n_nums,
        cards,
        input_size,
        hidden_size,
        num_layers,
        cat_emb_dim,
        time_dim,
        lat_dim,
    ):
        super().__init__()
        self.Emb = Embedding_data(
            input_size, cat_emb_dim, n_bins, n_cats, n_nums, cards
        )
        self.time_encode = nn.Sequential(
            nn.Linear(time_dim, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
        )

        self.encoder_mu = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder_logvar = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True
        )

        self.fc_mu = nn.Linear(hidden_size, lat_dim)
        self.fc_logvar = nn.Linear(hidden_size, lat_dim)

        # self.decoder_proj_in = nn.Linear(lat_dim, hidden_size)
        # self.decoder_mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(lat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.Emb_decoder = Embedding_data(
            input_size, cat_emb_dim, n_bins, n_cats, n_nums, cards
        )
        self.Emb_hidden_decoder = nn.Linear(input_size, hidden_size)

        self.channels = channels
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.cards = cards
        self.disc = self.n_bins + self.n_cats
        self.sigmoid = torch.nn.Sigmoid()

        self.bins_linear = nn.Linear(hidden_size, n_bins) if n_bins else None
        self.cats_linears = (
            nn.ModuleList([nn.Linear(hidden_size, card) for card in cards])
            if n_cats
            else None
        )
        self.nums_linear = nn.Linear(hidden_size, n_nums) if n_nums else None

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encoder(self, x):
        x = self.Emb(x)
        mu_z, _ = self.encoder_mu(x)
        logvar_z, _ = self.encoder_logvar(x)

        mu_z = self.fc_mu(mu_z)
        logvar_z = self.fc_logvar(logvar_z)
        emb = self.reparametrize(mu_z, logvar_z)

        return emb, mu_z, logvar_z

    def decompose_target_mask(self, mask):
        B, L, _ = mask.shape

        ###### Binary data type mask
        if self.n_bins is None or self.n_bins == 0:
            mask_bin = torch.zeros((B, L, 0), dtype=torch.float32).to(
                device
            )  # Return zero matrix
        else:
            mask_bin_temp = mask[:, :, : self.n_bins].to(device)
            mask_bin = torch.zeros_like(mask_bin_temp, dtype=torch.float32).to(device)
            mask_bin[mask_bin_temp == 0] = float("-100")

        ###### Categorical data type mask
        if self.n_cats is None or self.n_cats == 0:
            mask_cat = [
                torch.zeros((B, L, 0), dtype=torch.float32).to(device)
            ]  # Return a zero list
        else:
            mask_cat_temp = mask[:, :, self.n_bins : self.n_bins + self.n_cats].to(
                device
            )
            mask_cat = []
            for i in range(self.n_cats):
                mask_cat.append(torch.zeros(B * L, self.cards[i]).to(device))
                missing_indices = (mask_cat_temp[:, :, i].reshape(B * L) == 0).to(
                    device
                )
                mask_cat[i][missing_indices, 0] = float("100")
                mask_cat[i] = mask_cat[i].reshape(B, L, -1)

        ###### Numerical data type mask
        if self.n_nums is None or self.n_nums == 0:
            mask_num = torch.zeros((B, L, 0), dtype=torch.float32).to(
                device
            )  # Return zero matrix
        else:
            mask_num = mask[:, :, self.disc : self.disc + self.n_nums].to(device)

        return mask_bin, mask_cat, mask_num

    def decoder(self, latent_feature, mask=None, cond=None):
        decoded_outputs = dict()
        latent_feature = self.decoder_mlp(latent_feature)

        # latent_proj = self.decoder_proj_in(latent_feature)
        # latent_feature, _ = self.decoder_mha(latent_proj, latent_proj, latent_proj)

        mask_bin, mask_cat, mask_num = self.decompose_target_mask(mask)

        if cond is not None:
            cond_embedding = self.Emb_decoder(cond)
            latent_feature += self.Emb_hidden_decoder(cond_embedding)

        if self.bins_linear:
            decoded_outputs["bins"] = self.bins_linear(latent_feature) + mask_bin

        if self.cats_linears:
            decoded_outputs["cats"] = [
                linear(latent_feature) + mask_cat_el
                for linear, mask_cat_el in zip(self.cats_linears, mask_cat)
            ]

        if self.nums_linear:
            decoded_outputs["nums"] = (
                self.sigmoid(self.nums_linear(latent_feature)) * mask_num
            )

        return decoded_outputs

    def forward(self, x, mask, cond=None):
        emb, mu_z, logvar_z = self.encoder(x)
        outputs = self.decoder(emb, mask, cond)
        return outputs, emb, mu_z, logvar_z
