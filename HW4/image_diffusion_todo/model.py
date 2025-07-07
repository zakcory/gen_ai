from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    def get_loss(self, x0, class_label=None, noise=None):
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute noise matching loss.

        B = x0.shape[0]
        timestep = self.var_scheduler.uniform_sample_t(B, self.device) 

        # sample noise
        if noise is None:
            noise = torch.randn_like(x0)

        # diffuse
        x_t, epsilon = self.var_scheduler.add_noise(x0, timestep, noise)

        # predict noise
        noise_pred = self.network(x_t, timestep, class_label)

        # compute loss
        loss = F.mse_loss(noise_pred, epsilon)

        ######################
        return loss
    
    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 0.0,
    ):
        x_T = torch.randn([batch_size, 3, self.image_resolution, self.image_resolution]).to(self.device)

        do_classifier_free_guidance = guidance_scale > 0.0

        if do_classifier_free_guidance:

            ######## TODO ########
            # Assignment 2-3. Implement the classifier-free guidance.
            # Specifically, given a tensor of shape (batch_size,) containing class labels,
            # create a tensor of shape (2*batch_size,) where the first half is filled with zeros (i.e., null condition).
            assert class_label is not None
            assert len(class_label) == batch_size, f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"
            class_label = torch.cat([torch.zeros_like(class_label), class_label], dim=0)
            #######################

        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]
            if do_classifier_free_guidance:
                ######## TODO ########
                # Assignment 2. Implement the classifier-free guidance.

                # duplicate the current sample and timestep
                x_in = torch.cat([x_t, x_t], dim=0)                       

                # predict noise for both unconditional and conditional batches
                noise_pred_all = self.network(
                    x_in,
                    timestep=t.to(self.device),
                    class_label=class_label
                )

                # split and apply guidance
                noise_pred_uncond, noise_pred_cond = noise_pred_all.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                #######################
            else:
                noise_pred = self.network(
                    x_t,
                    timestep=t.to(self.device),
                    class_label=None,
                )

            x_t_prev = self.var_scheduler.step(x_t, t.to(self.device), noise_pred)

            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]

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
