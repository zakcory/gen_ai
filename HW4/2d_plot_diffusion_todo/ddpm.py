import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


class BaseScheduler(nn.Module):
    """
    Variance scheduler of DDPM.
    """

    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        mode: str = "linear",
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)


class DiffusionModule(nn.Module):
    """
    A high-level wrapper of DDPM and DDIM.
    If you want to sample data based on the DDIM's reverse process, use `ddim_p_sample()` and `ddim_p_sample_loop()`.
    """

    def __init__(self, network: nn.Module, var_scheduler: BaseScheduler):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        # For image diffusion model.
        return getattr(self.network, "image_resolution", None)


    @torch.no_grad()
    def q_sample(self, x0, t, noise=None):
        """
        sample x_t from q(x_t | x_0) of DDPM.

        Input:
            x0 (`torch.Tensor`): clean data to be mapped to timestep t in the forward process of DDPM.
            t (`torch.Tensor`): timestep
            noise (`torch.Tensor`, optional): random Gaussian noise. if None, randomly sample Gaussian noise in the function.
        Output:
            xt (`torch.Tensor`): noisy samples
        """
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_cumprod = self.var_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alpha_cumprod.sqrt()
        sqrt_one_minus_alphas_cumprod = (1.0 - alpha_cumprod).sqrt()

        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x0)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x0)

        xt = sqrt_alphas_cumprod_t * x0 \
             + sqrt_one_minus_alphas_cumprod_t * noise

        #######################

        return xt

    @torch.no_grad()
    def p_sample(self, xt, t):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.

        Input:
            xt (`torch.Tensor`): samples at arbitrary timestep t.
            t (`torch.Tensor`): current timestep in a reverse process.
        Ouptut:
            x_t_prev (`torch.Tensor`): one step denoised sample. (= x_{t-1})

        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute x_t_prev.
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device, dtype=torch.long)
        elif isinstance(t, torch.Tensor) and t.ndim == 0:
            t = t.unsqueeze(0).long().to(self.device)
        eps_theta = self.network(xt, t)

        beta_t = extract(self.var_scheduler.betas, t, xt)
        alpha_t = extract(self.var_scheduler.alphas, t, xt)
        alpha_prod_t = extract(self.var_scheduler.alphas_cumprod, t, xt)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alpha_prod_t)
        sqrt_recip_alphas = torch.sqrt( 1.0 / alpha_t )
        t_prev = (t - 1).clamp(min=0) # Just in case we don't want to have t < 0 in one of the entries
        alpha_prod_tm1 = extract(
            self.var_scheduler.alphas_cumprod,
            t_prev,
            xt
        )
        model_mean = (
                sqrt_recip_alphas *
                (xt - (beta_t / sqrt_one_minus_alphas_cumprod_t) * eps_theta)
        )
        posterior_variance = beta_t * (1.0 - alpha_prod_tm1) / (1.0 - alpha_prod_t)
        posterior_variance = posterior_variance.clamp(min=1e-20)


        mask = (t > 0).view(-1, *[1] * (xt.ndim - 1)) # Just for looking where t == 0 as in previous implementation
        x_t_prev = torch.where(
            mask,
            model_mean + torch.sqrt(posterior_variance) * torch.randn_like(xt),
            model_mean
        )

        ######################
        return x_t_prev


    @torch.no_grad()
    def p_sample_loop(self, shape):
        """
        The loop of the reverse process of DDPM.

        Input:
            shape (`Tuple`): The shape of output. e.g., (num particles, 2)
        Output:
            x0_pred (`torch.Tensor`): The final denoised output through the DDPM reverse process.
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # sample x0 based on Algorithm 2 of DDPM paper.

        xt = torch.randn(shape, device=self.device)
        for t in self.var_scheduler.timesteps.to(self.device): # Already reversed
            xt = self.p_sample(xt, t)
        x0_pred = xt
        ######################
        return x0_pred

    @torch.no_grad()
    def ddim_p_sample(self, xt, t, t_prev, eta=0.0):
        """
        One step denoising function of DDIM: x_t (τ_i) -> x_{τ_{i-1}}
        Returns x_t_prev.
        """
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device, dtype=torch.long)
        elif isinstance(t, torch.Tensor) and t.ndim == 0:
            t = t.unsqueeze(0)
        alpha_prod_t = extract(self.var_scheduler.alphas_cumprod, t, xt)
        alpha_prod_t_prev = (
            extract(self.var_scheduler.alphas_cumprod, t_prev, xt)
            if (t_prev >= 0).all() else
            torch.ones_like(alpha_prod_t)
        )

        eps_theta = self.network(xt, t)

        x0_pred = (xt - torch.sqrt(1 - alpha_prod_t) * eps_theta) / torch.sqrt(alpha_prod_t)

        sigma_t = (
                eta
                * torch.sqrt((1 - alpha_prod_t_prev) / (1 - alpha_prod_t))
                * torch.sqrt(1 - alpha_prod_t / alpha_prod_t_prev)
        )

        dir_coeff = torch.sqrt(
            torch.clamp(1 - alpha_prod_t_prev - sigma_t ** 2, min=0.0)
        )
        dir_xt = dir_coeff * eps_theta

        model_mean = torch.sqrt(alpha_prod_t_prev) * x0_pred + dir_xt

        if eta > 0.0:
            noise = torch.randn_like(xt)
            x_t_prev = model_mean + sigma_t * noise
        else:
            x_t_prev = model_mean

        return x_t_prev

    @torch.no_grad()
    def ddim_p_sample_loop(self, shape, num_inference_timesteps: int = 50, eta: float = 0.0):
        """
        DDIM reverse loop (T -> 0).

        Args:
            shape (Tuple[int]): output shape, e.g. (batch, C, H, W) or (N, 2).
            num_inference_timesteps (int): how many DDIM steps to take.
            eta (float): stochasticity coefficient (0 ⇒ deterministic).

        Returns:
            x0_pred (torch.Tensor): model’s reconstruction of x_0.
        """
        # ---------- prepare the DDIM time-schedule ----------
        step_ratio = self.var_scheduler.num_train_timesteps // num_inference_timesteps
        timesteps = (
            (np.arange(0, num_inference_timesteps) * step_ratio)
            .round()[::-1]
            .astype(np.int64)
        )
        timesteps = torch.from_numpy(timesteps).long()
        prev_timesteps = timesteps - step_ratio

        xt = torch.randn(shape, device=self.device)

        for t, t_prev in zip(timesteps, prev_timesteps):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            t_prev_tensor = torch.full((shape[0],), t_prev, device=self.device, dtype=torch.long)
            xt = self.ddim_p_sample(xt, t_tensor, t_prev_tensor, eta=eta)

        x0_pred = xt
        return x0_pred

    def compute_loss(self, x0):
        """
        The simplified noise matching loss corresponding Equation 14 in DDPM paper.

        Input:
            x0 (`torch.Tensor`): clean data
        Output:
            loss: the computed loss to be backpropagated.
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute noise matching loss.
        batch_size = x0.shape[0]
        t = (
            torch.randint(0, self.var_scheduler.num_train_timesteps, size=(batch_size,))
            .to(x0.device)
            .long()
        )
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        eps_hat = self.network(x_t, t)
        loss = F.mse_loss(eps_hat, noise)
        ######################
        return loss

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
        }
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
