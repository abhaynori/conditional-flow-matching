"""
2D-adapted version of PA-FM (Pseudo-Augmented Flow Matching)
Original: pafm_v2.py (designed for high-dimensional image data)
Adapted for: 2D toy distributions (e.g., 8 gaussians → moons)

Key Differences from pafm_v2.py:
1. Tensor shapes: [B, D] instead of [B, C, H, W]
2. Time sampling: [B, 1] instead of [B, 1, 1, 1]
3. Flattening operations: Direct usage instead of flatten(2) or flatten(1)
4. Model interface: Adapted for 2D (may return just output or tuple)
5. Removed image-specific components: projection loss, encoder features
"""

import math
import torch
import numpy as np
import torch.nn.functional as F


def mean_flat(x, temperature=1.0, **kwargs):
    """
    Take the mean over all non-batch dimensions.
    For 2D: x is [B, D], returns [B] (mean over dimension D)
    For images: x is [B, C, H, W], returns [B] (mean over C, H, W)
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def sum_flat(x, temperature=1.):
    """
    Take the sum over all non-batch dimensions.
    For 2D: x is [B, D], returns [B]
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

def mse_flat(x, y, temperature=1, **kwargs): 
    err = (x - y) ** 2
    return mean_flat(err)

def find_tau_for_target_ess(dists, target_ess=6.0, t_min=1e-3, iters=3):
    """
    dists: [B, M]: - squared distances / (2t^2)
    Returns: tau [B,] adaptive temperature
    """
    B, M = dists.shape
    # log domain for numerical stability
    log_tau = torch.zeros(B, device=dists.device)  # start tau=1
    low, high = -4.0, 4.0  # corresponds to tau in [exp(-4), exp(4)] ≈ [0.018, 54.6]
    # pdb.set_trace()
    for _ in range(iters):
        tau = torch.exp(log_tau).clamp_min(t_min)
        log_w = dists / (tau[:, None])           # [B, M]
        w = torch.softmax(log_w, dim=-1)
        ess = 1.0 / (w.pow(2).sum(dim=-1) + 1e-9)     # [B]
        # Update bounds
        too_sharp = ess < target_ess
        too_flat  = ess > target_ess
        # Move log_tau up/down
        log_tau = torch.where(too_sharp, log_tau + 0.5, log_tau)  # increase τ if ESS too low
        log_tau = torch.where(too_flat,  log_tau - 0.5, log_tau)  # decrease τ if ESS too high
    # pdb.set_trace()
    return torch.exp(log_tau)

def find_tau_for_target_ess2(
        dist2_over_t2,
        target_ess=8.0,
        tau_min=1e-3, 
        tau_max=1e3,
        bracket_iters=20,
        bisection_iters=50
    ):
    """
    dist2_over_t2: [B, M]  = ||x_t - (1-t) z||^2 / t^2  (no 1/(2tau) yet)
    t: [B]
    returns: alpha [B, M], tau [B]
    """
    B = dist2_over_t2.shape[0]
    # s: [B,M] = squared dists / (2 t^2); target_ess = 8.0
    tau_lo = torch.full([B], 1.0, device=dist2_over_t2.device)
    tau_hi = torch.full([B], 1.0, device=dist2_over_t2.device)

    def ess_at_tau(s, tau):
        # log-weights with max-subtraction
        lw = (s / tau[:, None])
        w  = torch.softmax(lw, dim=1)
        ess = 1.0 / (w.pow(2).sum(dim=1) + 1e-12)
        return ess

    # bracket: expand until ESS_lo < target < ESS_hi (per sample)
    for _ in range(bracket_iters):  # few expansions
        ess_lo = ess_at_tau(dist2_over_t2, tau_lo)
        ess_hi = ess_at_tau(dist2_over_t2, tau_hi)
        need_lo = ess_lo >= target_ess
        need_hi = ess_hi <= target_ess
        tau_lo[need_lo] = (tau_lo[need_lo] * 0.5).clamp_min(tau_min)
        tau_hi[need_hi] = (tau_hi[need_hi] * 2.0).clamp_max(tau_max)

    # bisection
    tau = (tau_lo + tau_hi) / 2
    for _ in range(bisection_iters):  # ~10 iters gives ~1e-3 relative accuracy
        ess = ess_at_tau(dist2_over_t2, tau)
        go_hi = ess < target_ess
        if go_hi.sum() == 0 or (~go_hi).sum() == 0:
            break  # all converged
        tau_lo = torch.where(go_hi, tau, tau_lo)
        tau_hi = torch.where(go_hi, tau_hi, tau)
        tau = (tau_lo + tau_hi) / 2
    # use tau
    return tau



class PAFM2DLoss:
    """
    2D-adapted Pseudo-Augmented Flow Matching Loss
    Perfectly emulates pafm_v2.py logic but for 2D inputs.
    
    Core Algorithm (unchanged from pafm_v2.py):
    1. Sample time t ~ Uniform(0,1) or LogNormal
    2. Create interpolated samples: x_t = α(t)·x_0 + σ(t)·noise
    3. Compute FM loss: ||v_θ(x_t,t) - (x_1 - x_0)||²
    4. Compute pseudo-targets from batch: v_pseudo^j = (x_t^i - x_1^j) / t
    5. Weight pseudo-targets by p(x_t^i|x_1^j) and p(y|z)
    6. Combined loss: λ·FM_loss + (1-λ)·weighted_pseudo_loss
    """
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            null_class_idx=None,
            fm_interpolant=0.0,
            pt_xt_y_method="l2",
            pt_xt_y_tau=0.05,  # Matches pafm_v2.py default
            p_y_z_tau=0.1,
            **kwargs
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.null_class_idx = null_class_idx
        # Augmentation settings
        self.fm_interpolant = fm_interpolant
        self.pt_xt_y_tau = pt_xt_y_tau
        self.p_y_z_tau = p_y_z_tau
        self.pt_xt_y_method = pt_xt_y_method
        print(f"Setting fm loss to: {self.fm_interpolant} and the PA-FM loss to {1 - self.fm_interpolant}")
        print(f"Using p_t(x_t^i|z^j) Temp of: {self.pt_xt_y_tau} and p(y|z) Temp of: {self.p_y_z_tau}")
        print(f"Using pt_xt_y_method: {self.pt_xt_y_method}")

    def interpolant(self, t):
        """
        Compute interpolation path and its derivatives.
        IDENTICAL to pafm_v2.py - dimension-agnostic operation.
        
        Args:
            t: [B, 1] time values in [0, 1]
        Returns:
            alpha_t, sigma_t: interpolation coefficients
            d_alpha_t, d_sigma_t: time derivatives (for velocity target)
        """
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
            raise NotImplementedError(f"path_type {self.path_type} not implemented")

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def __call__(
            self, 
            model, 
            images,  # [B, D] - keeping 'images' name for consistency with pafm_v2.py
            model_kwargs=None, 
            zs=None,  # Not used in 2D, kept for API compatibility
            label_distribution=None, 
            supervise_images=None,  # [B, D] - target samples
            batch_offset=0,
            **kwargs
        ):
        """
        Compute PA-FM loss for 2D data.
        Perfectly emulates pafm_v2.py structure but adapted for 2D tensor shapes.
        
        Args:
            model: Neural network, model(x, t, y=...) → velocity [B, D] (or tuple)
            images: [B, D] source distribution samples (x0)
            supervise_images: [B, D] target distribution samples (x1)
            model_kwargs: dict with 'y' labels [B] (optional)
            label_distribution: [B, num_classes] class distribution (optional)
            batch_offset: int, offset for diagonal indexing
            
        Returns:
            denoising_loss: dict with loss components
            proj_loss: 0 (not used in 2D, kept for API compatibility)
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        B, D = images.shape  # B is minibatch size
        B_full = supervise_images.shape[0]  # Full batch size (pool)
        device = images.device
        dtype = images.dtype
        
        # Sample timesteps - adapted shape for 2D
        if self.weighting == "uniform":
            time_input = torch.rand((B, 1), device=device, dtype=dtype)
        elif self.weighting == "lognormal":
            rnd_normal = torch.randn((B, 1), device=device, dtype=dtype)
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
        else:
            raise ValueError(f"weighting {self.weighting} not supported")
        


        time_input = time_input.to(device=images.device, dtype=images.dtype)

        # Create noisy samples via interpolation
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
        
        model_target = d_alpha_t * images + d_sigma_t * noises  # [B, D]
        model_input = alpha_t * images + sigma_t * noises  # [B, D]
        
        # Get model prediction - adapted for 2D models
        # 2D models typically take concatenated [x, t] input
        model_input_with_time = torch.cat([model_input, time_input], dim=-1)  # [B, D+1]
        model_output = model(model_input_with_time)  # [B, D]
        
        # Standard Flow Matching loss
        fm_loss = mean_flat((model_output - model_target) ** 2)  # [B]
        
        # Compute pseudo-targets from batch
        image_targets = supervise_images[None]  # [1, B, D]
        pseudo_targets = (model_input[:, None] - image_targets) / time_input[:, None]  # [B, B, D]
        # pseudo_loss = (model_output[:, None] - pseudo_targets).pow(2).mean(-1)  # [B, B]
        pseudo_loss = (model_output[:, None] - pseudo_targets).pow(2).flatten(2).mean(-1)  # [B, B]

        # Compute importance weights p(x_t^i | x_1^j)
        if self.pt_xt_y_method == "l2":
            alpha_t_xt_z = - (
                # (model_input[:, None] - (1 - time_input[:, None]) * image_targets).pow(2).sum(-1) /
                # (2 * time_input.pow(2))  # [B, B]
                (model_input[:, None] - (1 - time_input[:, None]) * image_targets).flatten(2).pow(2).sum(-1) /
                (2 * (time_input.reshape(-1, 1).pow(2)))
            )
        elif self.pt_xt_y_method == "cosine":
            time_ = time_input.reshape(-1, 1)
            unit_zi = F.normalize(images, dim=-1)  # [B, D]
            unit_zj = F.normalize(supervise_images, dim=-1)  # [B, D]
            cos_sim = (unit_zi[:, None] * unit_zj[None]).sum(-1)  # [B, B]
            cos_dist = 1 - cos_sim  # [B, B]
            alpha_t_xt_z = -((1 - time_).pow(2) / (time_.pow(2))) * cos_dist  # [B, B]
        else:
            raise NotImplementedError(f"pt_xt_y_method {self.pt_xt_y_method} not implemented")
        
        alpha_t_xt_z = F.log_softmax(alpha_t_xt_z * self.pt_xt_y_tau, dim=-1)  # [B, B]
        pt_xt_z = alpha_t_xt_z.exp()  # [B, B]
        pt_xt_z_ess = 1 / (pt_xt_z.pow(2).sum(dim=-1) + 1e-9)
        
        # Incorporate label distribution p(y|z)
        if label_distribution is not None and "y" in model_kwargs:
            labels = model_kwargs["y"]
            alpha_y_z = label_distribution.index_select(dim=1, index=labels).T.clamp_min(1e-12)  # [B, B]
            if self.null_class_idx is not None:
                alpha_y_z[labels == self.null_class_idx] = 1.0
            # Note: p_y_z_tau multiplication is commented out in pafm_v2.py
            p_y_z = alpha_y_z.log()  # * self.p_y_z_tau
        else:
            # Unconditional case - shape must be [B_mini, B_full]
            p_y_z = torch.zeros(B, B_full, device=device, dtype=dtype)
            labels = torch.zeros(B, dtype=torch.long, device=device)
        
        # Combine scores and compute final weights
        combined_scores = alpha_t_xt_z + p_y_z  # [B_mini, B_full]
        combined_scores = F.softmax(combined_scores, dim=-1)  # [B_mini, B_full]
        combined_ess = 1 / (combined_scores.pow(2).sum(dim=-1) + 1e-9)
        
        # Compute weighted pseudo-target loss
        pafm_loss = (combined_scores * pseudo_loss)  # [B_mini, B_full]
        
        # Extract "real pair" loss (diagonal within the relevant slice)
        # For minibatch processing: real pairs are at indices [batch_offset : batch_offset + B]
        # real_pair_indices = torch.arange(B, device=device) + batch_offset
        # if batch_offset + B <= B_full:
        #     # Extract the diagonal from the relevant slice
        #     approx_fm_loss = pafm_loss[torch.arange(B, device=device), real_pair_indices].mean()
        # else:
        #     # If batch_offset puts us beyond the full batch, just use mean
        #     approx_fm_loss = pafm_loss.mean()
        
        # aux_loss = pafm_loss.sum(dim=-1) - approx_fm_loss
        approx_fm_loss = pafm_loss[:, batch_offset:].diag().mean()
        aux_loss = pafm_loss.sum(dim=-1)
        
        # Final combined loss
        loss = fm_loss * self.fm_interpolant + (1 - self.fm_interpolant) * pafm_loss.sum(dim=-1)  # [B]
        
        # Compute weighted target for alignment metric
        combined_target = (combined_scores[:, :, None] * pseudo_targets).sum(1)  # [B, D]
        
        # # Compute real pair weights (diagonal within relevant slice)
        # if batch_offset + B <= B_full:
        #     real_pair_weights = combined_scores[torch.arange(B, device=device), real_pair_indices].mean()
        # else:
        #     real_pair_weights = combined_scores.mean()
        
        # Return loss dictionary
        denoising_loss = {
            "loss": loss,
            "approx_flow_loss": approx_fm_loss,
            "fm_loss": fm_loss,
            "augmented_loss": aux_loss,
            # "weights_real": real_pair_weights,
            # "weights_augmented": (combined_scores.sum(dim=1) - real_pair_weights * B).mean() / (B_full - 1),
            "ess_pt_xt_z": pt_xt_z_ess[labels != self.null_class_idx].mean(),
            "ess_combined": combined_ess[labels != self.null_class_idx].mean(),
            # "sigma_0": sigma_0,
            "flow_alignment": (F.normalize(model_target.flatten(1), dim=-1) * F.normalize(combined_target.flatten(1), dim=-1)).sum(-1).mean() # [B,C*H*W]x[B,C*H*W] -> [B]
        }
        
        # No projection loss in 2D (was for image encoder features)
        proj_loss = 0.
        
        return denoising_loss, proj_loss
