"""
2D version of PA-FM loss for toy datasets.
Heavily modified from REPA: https://github.com/sihyun-yu/REPA/blob/main/loss.py
"""

import pdb
import math
import torch
import numpy as np
import torch.nn.functional as F


def mean_flat(x, temperature=1.0, **kwargs):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def sum_flat(x, temperature=1.):
    """
    Take the sum over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))


def mse_flat(x, y, temperature=1, **kwargs):
    err = (x - y) ** 2
    return mean_flat(err)


class PAFMLossV2_2D:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            fm_interpolant=0.0,
            pt_xt_y_method="l2",
            pt_xt_y_tau=0.05,
            p_y_z_tau=0.1,
            **kwargs
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        # Augmentation settings
        self.fm_interpolant = fm_interpolant
        self.pt_xt_y_tau = pt_xt_y_tau
        self.p_y_z_tau = p_y_z_tau
        self.pt_xt_y_method = pt_xt_y_method
        print(f"Setting fm loss to: {self.fm_interpolant} and the PA-FM loss to {1 - self.fm_interpolant}")
        print(f"Using p_t(x_t^i|z^j) Temp of: {self.pt_xt_y_tau} and p(y|z) Temp of: {self.p_y_z_tau}")
        print(f"Using pt_xt_y_method: {self.pt_xt_y_method}")

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t = 1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def __call__(self, model, data, supervise_data=None, batch_offset=0, **kwargs):
        """
        Args:
            model: Neural network that takes (x_t, t) and outputs velocity
            data: [B, 2] tensor of 2D points
            supervise_data: [K, 2] tensor of supervision points (default: use data itself)
            batch_offset: offset for diagonal extraction when using minibatches
        """
        # Use data as supervision if not provided
        if supervise_data is None:
            supervise_data = data
        
        B = data.shape[0]
        K = supervise_data.shape[0]
        
        # Sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((B, 1), device=data.device, dtype=data.dtype)
        elif self.weighting == "lognormal":
            # Sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((B, 1), device=data.device, dtype=data.dtype)
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
        
        # Generate noise
        noises = torch.randn_like(data)
        
        # Get interpolant values
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
        
        # Compute model input and target
        model_target = d_alpha_t * data + d_sigma_t * noises  # [B, 2]
        model_input = alpha_t * data + sigma_t * noises  # [B, 2]
        
        # Get model prediction
        model_output = model(model_input, time_input.squeeze(-1))  # [B, 2]
        
        # Standard flow matching loss
        fm_loss = ((model_output - model_target) ** 2).mean(dim=1)  # [B]
        
        # Compute pseudo targets for each supervision point
        image_targets = supervise_data.unsqueeze(0)  # [1, K, 2]
        pseudo_targets = (model_input.unsqueeze(1) - image_targets) / time_input.unsqueeze(1)  # [B, K, 2]
        pseudo_loss = ((model_output.unsqueeze(1) - pseudo_targets) ** 2).mean(dim=-1)  # [B, K]
        
        # Compute p_t(x_t^i | z^j) weights
        if self.pt_xt_y_method == "l2":
            # L2 distance in data space
            dist_sq = ((model_input.unsqueeze(1) - (1 - time_input.unsqueeze(1)) * image_targets) ** 2).sum(dim=-1)  # [B, K]
            alpha_t_xt_z = -dist_sq / (2 * time_input.pow(2))  # [B, K]
        elif self.pt_xt_y_method == "cosine":
            # Cosine similarity approach
            time_ = time_input  # [B, 1]
            scaled_noise = (time_ / (1 - time_)) * noises  # [B, 2]
            unit_zi = F.normalize(data + scaled_noise, dim=-1)  # [B, 2]
            unit_zj = F.normalize(supervise_data, dim=-1)  # [K, 2]
            cos_sim = (unit_zi.unsqueeze(1) * unit_zj.unsqueeze(0)).sum(dim=-1)  # [B, K]
            cos_dist = 1 - cos_sim  # [B, K]
            alpha_t_xt_z = -((1 - time_).pow(2) / time_.pow(2)) * cos_dist  # [B, K]
        else:
            raise NotImplementedError(f"pt_xt_y_method {self.pt_xt_y_method} not implemented")
        
        # Apply temperature and softmax
        alpha_t_xt_z = F.log_softmax(alpha_t_xt_z / self.pt_xt_y_tau, dim=-1)  # [B, K]
        pt_xt_z = alpha_t_xt_z.exp()  # [B, K]
        pt_xt_z_ess = 1 / (pt_xt_z.pow(2).sum(dim=-1) + 1e-9)  # [B]
        
        # For 2D data without class labels, use uniform p(y|z)
        # You can modify this if you have class information
        p_y_z = torch.zeros_like(alpha_t_xt_z)  # Uniform in log space
        
        # Combine scores
        combined_scores = alpha_t_xt_z + p_y_z  # [B, K]
        combined_scores = F.softmax(combined_scores, dim=-1)  # [B, K]
        combined_ess = 1 / (combined_scores.pow(2).sum(dim=-1) + 1e-9)  # [B]
        
        # Compute PA-FM loss
        pafm_loss = (combined_scores * pseudo_loss).sum(dim=-1)  # [B]
        
        # Approximate standard FM loss from PA-FM
        if batch_offset + B <= K:
            approx_fm_loss = (combined_scores[:, batch_offset:batch_offset+B].diagonal() * 
                            pseudo_loss[:, batch_offset:batch_offset+B].diagonal()).mean()
        else:
            approx_fm_loss = pafm_loss.mean()
        
        # Auxiliary loss (contribution from augmented samples)
        aux_loss = pafm_loss - approx_fm_loss
        
        # Combined loss
        loss = fm_loss * self.fm_interpolant + (1 - self.fm_interpolant) * pafm_loss
        
        # Compute combined target for analysis
        combined_target = (combined_scores.unsqueeze(-1) * pseudo_targets).sum(dim=1)  # [B, 2]
        
        # Compute metrics
        loss_dict = {
            "loss": loss.mean(),
            "approx_flow_loss": approx_fm_loss,
            "fm_loss": fm_loss.mean(),
            "augmented_loss": aux_loss.mean(),
            "weights_real": combined_scores[:, batch_offset:batch_offset+min(B, K-batch_offset)].diagonal().mean() if batch_offset + B <= K else 0.0,
            "weights_augmented": (combined_scores.sum(dim=1) - combined_scores[:, batch_offset:batch_offset+min(B, K-batch_offset)].diagonal()).mean() if batch_offset + B <= K else combined_scores.sum(dim=1).mean(),
            "ess_pt_xt_z": pt_xt_z_ess.mean(),
            "ess_combined": combined_ess.mean(),
            "flow_alignment": (F.normalize(model_target, dim=-1) * F.normalize(combined_target, dim=-1)).sum(-1).mean(),
            "combined_target_norm": combined_target.norm(dim=-1).mean(),
            "flow_target_norm": model_target.norm(dim=-1).mean(),
        }
        
        return loss_dict, None  # Return None for proj_loss to match original API


class SimplePAFMWrapper:
    """
    Simple wrapper that only returns the loss value for easy integration.
    """
    def __init__(self, **kwargs):
        self.loss_fn = PAFMLossV2_2D(**kwargs)
    
    def __call__(self, model, data, supervise_data=None):
        loss_dict, _ = self.loss_fn(model, data, supervise_data)
        return loss_dict["loss"]
