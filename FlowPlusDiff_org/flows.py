import math
from denoising_diffusion_pytorch import Unet1D
import torch
import torch.nn as nn
from torch.distributions import Normal
from embed import DataEmbedding,DataEmbeddingcond
import torch.nn.functional as F

class Transformnont(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, nhead=4, num_layers=4):
        super().__init__()

        self.input_proj = DataEmbedding(in_channels//2, hidden_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Linear(hidden_channels//2, in_channels)
        )
        self.asig = AffineSigmoid()

    def forward(self, x):
        _,_,fea=x.shape
        x=x[:,:,:fea//2]
        h = self.input_proj(x)
        h = self.transformer(h)
        t = self.head(h)
        t = self.asig.as_sigmoid(t)
        return t


class Transform(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, nhead=4, num_layers=4):
        super().__init__()

        self.input_proj = DataEmbeddingcond(in_channels//2, hidden_channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Linear(hidden_channels//2, in_channels)
        )
        self.asig = AffineSigmoid()

    def forward(self, x,t):
        b,l,fea=x.shape

        x=x[:,:,:fea//2]
        h = self.input_proj(x,t)
        h = self.transformer(h)
        t = self.head(h)
        t = self.asig.as_sigmoid(t)
        return t
    
class Transformst(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, nhead=4, num_layers=4):
        super().__init__()

        self.input_proj = DataEmbedding(in_channels, hidden_channels)
        self.in_channels=in_channels
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=nhead,
            batch_first=True
        )
        

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.scale = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Linear(hidden_channels//2, in_channels)
        )
        self.bias = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Linear(hidden_channels//2, in_channels)
        )

    def forward(self, x):
        h0 = self.input_proj(x)
        h = self.transformer(h0)
        h=h+h0
        t = self.bias(h)
        s = self.scale(h)
        log_s = torch.tanh(s)

        return t, log_s

    
class AffineSigmoid(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps  # stability for inverse

    def forward(self, z):
        """
        Forward transform: z -> y
        Returns:
            y: bounded output in (-1,1)
            logdet: log absolute Jacobian
        """

        sigma = torch.sigmoid(z)
        y = 2.0 * sigma - 1.0

        logdet = (
            math.log(2.0)
            - F.softplus(-z)
            - F.softplus(z)
        )

        logdet = logdet.sum(dim=-1, keepdim=True)
        return y, logdet


    def inverse(self, y):
        """
        Inverse transform: y -> z
        """
        y = y.clamp(-1 + self.eps, 1 - self.eps)  # avoid numerical issues
        z = torch.log((y + 1) / (1 - y))
        return z

    def logdet_inverse(self, y):
        """
        Optional: compute logdet of inverse (just -forward logdet)
        """
        z = self.inverse(y)
        _, logdet_fwd = self.forward(z)
        return -logdet_fwd
    
    def as_sigmoid(self, y):
        sigma = torch.sigmoid(y)
        y = 2.0 * sigma - 1.0
        return y

class TanhScale(nn.Module):
    def __init__(self, scale=3.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        return torch.tanh(x) * self.scale

class AugmentedBlock(nn.Module):
    def __init__(self, x_dim, mask, n_neurons=128, nhead=4, flowattlayer=4, window=128, last=False):
        """
        Intialise the block
        """
        super(AugmentedBlock, self).__init__()
        self.last = last
        self.trans=Transformst(x_dim, n_neurons, nhead, flowattlayer)
        self.asig=AffineSigmoid()
    
        self.mask=mask
        self.tanh=nn.Tanh()

    def forward(self, x, mode='forward'):

            if mode == 'forward':
                t ,s=self.trans(x * self.mask)
                y = self.mask * x + (1 - self.mask) * (x * torch.exp(s) + t)
                logdet = ((1 - self.mask) * s).sum(dim=(-1), keepdim=True)
                if self.last:
                    y,logdetraw= self.asig.forward(y)
                    logdet += logdetraw
                return y, logdet
            elif mode == 'reverse':  # inverse
                if self.last:
                    # clamp to avoid log(0)
                    x=self.asig.inverse(x)
                    logdet=0
                else:
                    logdet = 0.0
                t ,s=self.trans(x * self.mask)
                x_out = self.mask * x + (1 - self.mask) * ((x - t) * torch.exp(-s))
                logdet -= torch.sum((1 - self.mask) * s, dim=(-1), keepdim=True)
           
                return x_out, logdet
            
def extract(a, t, x_shape):
    # a: [T], t: [B], x_shape: (B, L, F) or similar
    out = a[t]  # shape [B]
    return out.view(-1, *((1,) * (len(x_shape) - 1)))  # [B,1,1...] broadcastable


def create_masks_1d(L, F, n_layers, ratio=0.25, device=None):
    k = max(1, int(round(F * ratio)))
    masks = []
    step = F // k   # spacing between active features

    for i in range(n_layers):
        mask_1d = torch.zeros(F)
        idx = [(i + j * step) % F for j in range(k)]
        mask_1d[idx] = 1.0
        mask = mask_1d.view(1, 1, F).repeat(1, L, 1)
        masks.append(mask.to(device))
    return masks

class AugmentedSequential(nn.Sequential):
    def __init__(self, *flow_blocks, d_model,n_flow_layers,nhead,flowattlayer,diff_layers,n_times,window,Fea,ratio,betamin,betamax,U_d,only_transformer,diffst,diffusion_conditioning,infer_samples=30,device=None):
        super().__init__(*flow_blocks)
    
        masks = create_masks_1d(window,Fea, n_flow_layers,ratio, device)
        self.diffst=diffst
        self.modeles = nn.ModuleList([
            AugmentedBlock(Fea, m, d_model, nhead, flowattlayer, window, last=False) for m in masks[:-1]  # all except last
        ]+[
            AugmentedBlock(Fea, masks[-1], d_model, nhead, flowattlayer, window, last=True)       # last block with special flag
        ])

        # --- Diffusion params ---
        self.d_model=d_model
        self.diffusion_conditioning=diffusion_conditioning
        self.only_transformer=only_transformer
        if self.only_transformer:
            self.transformer = Transformnont(Fea,d_model, nhead,diff_layers)
        else:
            self.transformer = Transform(Fea,d_model, nhead,diff_layers)
        if self.diffusion_conditioning:
            unetfea=Fea*2
        else:
            unetfea=Fea

        self.denoiser = DenoiserWithScale(
            Unet1D(
                dim=d_model,
                dim_mults=U_d,
                channels=unetfea,
                out_dim=Fea
            )
        )
        self.tanh=nn.Tanh()
        self.T = n_times
        self.infer_samples = infer_samples

        beta_1, beta_T = betamin, betamax

        # linear beta schedule
        betas = torch.linspace(beta_1, beta_T, n_times)
        self.asig=AffineSigmoid()
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt(1.0 - alpha_bars)
        )
        self.log_var = nn.Parameter(torch.zeros(1))
        
        
    def forward(self, y, mode='forward'):
        """
        Forward or backward pass through the flows
        """
        logdet_tot = torch.zeros(y.size(0), y.size(1), 1, device=y.device)

        if mode == 'forward':
            for module in self.modeles:
                y, log_J = module(y, mode)
                logdet_tot += log_J
        else: 
            for module in reversed(self.modeles):
                y, log_J = module(y, mode)
                logdet_tot += log_J

        return y, logdet_tot


    def infer(self, feature_input, score_type="l2"):
        """
        Inference with different scoring methods.
        score_type options:
        - "l2": Simple L2 distance to mean (fastest)
        - "lat-ll": Latent log-likelihood (requires variance)
        - "nf-ll": Negative flow log-likelihood (full likelihood)
        """
        B,L,Fea = feature_input.shape
        device = feature_input.device 
        
        z_flow, logdet_tot= self(feature_input, mode='forward')
        if self.only_transformer:
            with torch.inference_mode():
                mu = self.transformer(feature_input)
                score = torch.sum((z_flow - mu)**2, dim=-1)
                feature_recon,_= self(mu, mode='reverse')
                feature_reconstd=torch.ones_like(z_flow)
            return feature_recon,feature_reconstd,score
        elif self.only_transformer==None:
            with torch.inference_mode():
                score = torch.sum((z_flow)**2, dim=-1)
                feature_recon,_= self(torch.zeros_like(z_flow), mode='reverse')
                feature_reconstd=torch.ones_like(z_flow)
            return feature_recon,feature_reconstd,score
        else:
            with torch.inference_mode():
                n = self.infer_samples
                flat_latent = torch.randn(B*n,L, Fea, device=device)
                feature_input_exp = feature_input.unsqueeze(1).expand(-1, n, -1, -1).contiguous()
                ddim_steps = torch.linspace(self.T-1, 0, self.diffst, device=device).long()
                flat_latent=self.generate(ddim_steps,feature_input_exp.view(B * n, L, Fea),flat_latent,device,feature_input_exp.shape,self.diffusion_conditioning,1)
                flat_latent = self.asig.as_sigmoid(flat_latent)
                feature_reconraw,_= self(flat_latent, mode='reverse')
                feature_recon=feature_reconraw.view(B, n, L, Fea).mean(dim=1)
                feature_reconstd=feature_reconraw.view(B, n, L, Fea).std(dim=1)
                z_diff = flat_latent.view(B, n, L, Fea)
                mu = z_diff.mean(1)  # (B, L, Fea)
                sigma = z_diff.std(dim=1) # (B, L, Fea) - add small value for stability
                
                # Compute score based on selected type
                if score_type in ["l2"]:
                    score = torch.sum((z_flow - mu)**2, dim=-1)
                elif score_type in ["lat-ll"]:
                    # Case 2: Latent Log-Likelihood (lat-LL)
                    dist = Normal(mu, sigma)
                    log_prob = dist.log_prob(z_flow).sum(-1)
                    score = -log_prob
                elif score_type in ["nf-ll"]:
                    # Case 3: Negative Flow Log-Likelihood (NF-LL)
                    dist = Normal(mu, sigma)
                    log_prob = dist.log_prob(z_flow).sum(-1)
                    score = -(logdet_tot.squeeze(-1) + log_prob)
                else:
                    score = torch.sum((z_flow - mu)**2, dim=-1)

        return feature_recon,feature_reconstd,score,z_flow,mu
    
    def generate(self,ddim_steps,contextin,flat_latent,device,shape,diffusion_conditioning,eta):
        B,n,L,Fea=shape
        for i in range(len(ddim_steps) - 1):
            t = ddim_steps[i]
            t_next = ddim_steps[i + 1]
            t_tensor = torch.full((B * n,), t, device=device, dtype=torch.long)
            denoiser_input = flat_latent.permute(0, 2, 1)  # (B*n, F, L)

            if diffusion_conditioning:
                context = self.transformer(contextin,t_tensor).permute(0,2,1)
                denoiser_input = torch.cat((denoiser_input, context), dim=-2)

            eps_pred = self.denoiser(denoiser_input, t_tensor).permute(0, 2, 1)

            alpha_bar_t = extract(self.alpha_bars, t_tensor, flat_latent.shape)
            t_next_tensor = torch.full((B * n,), t_next, device=device, dtype=torch.long)
            alpha_bar_next = extract(self.alpha_bars, t_next_tensor, flat_latent.shape)

            x0 = (flat_latent - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

            # compute sigma for stochasticity
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_next) / (1 - alpha_bar_t) *
                (1 - alpha_bar_t / alpha_bar_next)
            )

            z_noise = torch.randn_like(flat_latent)

            flat_latent = (
                torch.sqrt(alpha_bar_next) * x0 +
                torch.sqrt(1 - alpha_bar_next - sigma**2) * eps_pred +
                sigma * z_noise
            )
        return flat_latent
    
    def flow(self, feature):
        z_0,logdet_tot= self(feature, mode='forward')
        return z_0,logdet_tot
    
    def diffusion(self, z_0, feature):
        B, L, Fea = feature.shape
        device = feature.device
        if self.only_transformer is True:
            mu = self.transformer(feature)
            sigma = torch.exp(0.5 * self.log_var)
            sigma=sigma.clamp(0.1,5)
        elif self.only_transformer is False:
            t = torch.randint(0, self.T, (B,), device=device)
            epsilon = torch.randn((B, L, Fea), device=device)
            sqrt_alpha_bar = extract(self.sqrt_alpha_bars, t, (B, L, Fea))
            sqrt_one_minus_alpha_bar = extract(
                self.sqrt_one_minus_alpha_bars, t, (B, L, Fea)
            )
            z_t = z_0 * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
            denoiser_input = z_t.permute(0, 2, 1)  # (B, F, L)
            if self.diffusion_conditioning:
                context = self.transformer(feature, t)
                denoiser_input = torch.concat((denoiser_input, context.permute(0, 2, 1)), -2)
            eps_pred = self.denoiser(denoiser_input, t).permute(0, 2, 1).float()
            mu = (z_t - eps_pred * sqrt_one_minus_alpha_bar) / (sqrt_alpha_bar)
            sigma = torch.exp(0.5 * self.log_var)
            sigma=sigma.clamp(0.1,5)
        else:
            # Fallback: identity
            mu = torch.zeros_like(feature)
            sigma = 1.0

        return mu, sigma

class DenoiserWithScale(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    
    def forward(self, x, t):
        out = self.unet(x, t)
        return out # clamp to prevent extreme values

