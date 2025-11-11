## Code Cells to Add to the Notebook

Add these cells after the existing training examples to test the PA-FM loss function:

```python
# Cell 1: Import the PA-FM loss function
from pafm_2d import PAFMLossV2_2D, SimplePAFMWrapper

# Cell 2: Setup PA-FM training
print("Training with PA-FM Loss")

# Initialize PA-FM loss
pafm_loss_fn = PAFMLossV2_2D(
    prediction='v',
    path_type="linear",
    weighting="uniform",
    fm_interpolant=0.5,  # 50% standard FM, 50% PA-FM
    pt_xt_y_method="l2",
    pt_xt_y_tau=0.05,
    p_y_z_tau=0.1
)

# Create a new model for PA-FM
model_pafm = MLP(dim=2, time_varying=True)
optimizer_pafm = torch.optim.Adam(model_pafm.parameters())

# Cell 3: Training loop for PA-FM
start = time.time()
losses_pafm = []

for k in range(20000):
    optimizer_pafm.zero_grad()
    
    # Sample source and target data
    x0 = sample_8gaussians(batch_size)
    x1 = generate_moons(batch_size)
    
    # Compute PA-FM loss
    loss_dict, _ = pafm_loss_fn(model_pafm, x1, supervise_data=x1)
    
    loss = loss_dict["loss"]
    loss.backward()
    optimizer_pafm.step()
    
    losses_pafm.append(loss.item())
    
    if (k + 1) % 5000 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():.3f} time {(end - start):.2f}")
        print(f"  FM loss: {loss_dict['fm_loss'].item():.3f}")
        print(f"  Augmented loss: {loss_dict['augmented_loss'].item():.3f}")
        print(f"  ESS combined: {loss_dict['ess_combined'].item():.3f}")
        print(f"  Flow alignment: {loss_dict['flow_alignment'].item():.3f}")
        
        # Generate and plot samples
        node = NeuralODE(model_pafm, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        with torch.no_grad():
            traj = node.trajectory(
                sample_8gaussians(1024),
                t_span=torch.linspace(0, 1, 100),
            )
            plt.figure(figsize=(6, 6))
            plt.scatter(traj[-1, :, 0].cpu().numpy(), traj[-1, :, 1].cpu().numpy(), s=10, alpha=0.8, c="black")
            plt.legend()
            plt.xticks([])
            plt.yticks([])
            plt.title(f"PA-FM at step {k+1}")
            plt.show()
        
        start = time.time()

torch.save(model_pafm, f"{savedir}/pafm-moons.pt")

# Cell 4: Compare different fm_interpolant values
print("\nComparing different fm_interpolant values:")

interpolant_values = [0.0, 0.25, 0.5, 0.75, 1.0]
results_comparison = {}

for fm_interp in interpolant_values:
    print(f"\nTesting fm_interpolant={fm_interp}")
    
    # Initialize model and loss
    model_test = MLP(dim=2, time_varying=True)
    optimizer_test = torch.optim.Adam(model_test.parameters())
    
    loss_fn_test = PAFMLossV2_2D(
        prediction='v',
        path_type="linear",
        weighting="uniform",
        fm_interpolant=fm_interp,
        pt_xt_y_method="l2",
        pt_xt_y_tau=0.05,
        p_y_z_tau=0.1
    )
    
    # Train for fewer iterations for comparison
    final_losses = []
    for k in range(5000):
        optimizer_test.zero_grad()
        
        x0 = sample_8gaussians(batch_size)
        x1 = generate_moons(batch_size)
        
        loss_dict, _ = loss_fn_test(model_test, x1, supervise_data=x1)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer_test.step()
        
        if k >= 4000:  # Collect last 1000 losses
            final_losses.append(loss.item())
    
    results_comparison[fm_interp] = {
        'mean_loss': np.mean(final_losses),
        'std_loss': np.std(final_losses)
    }
    
    print(f"  Final loss: {np.mean(final_losses):.3f} ± {np.std(final_losses):.3f}")

# Cell 5: Compare L2 vs Cosine methods for pt_xt_y
print("\nComparing pt_xt_y methods:")

methods = ["l2", "cosine"]
results_methods = {}

for method in methods:
    print(f"\nTesting method={method}")
    
    model_test = MLP(dim=2, time_varying=True)
    optimizer_test = torch.optim.Adam(model_test.parameters())
    
    loss_fn_test = PAFMLossV2_2D(
        prediction='v',
        path_type="linear",
        weighting="uniform",
        fm_interpolant=0.5,
        pt_xt_y_method=method,
        pt_xt_y_tau=0.05,
        p_y_z_tau=0.1
    )
    
    final_losses = []
    final_ess = []
    
    for k in range(5000):
        optimizer_test.zero_grad()
        
        x0 = sample_8gaussians(batch_size)
        x1 = generate_moons(batch_size)
        
        loss_dict, _ = loss_fn_test(model_test, x1, supervise_data=x1)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer_test.step()
        
        if k >= 4000:
            final_losses.append(loss.item())
            final_ess.append(loss_dict['ess_combined'].item())
    
    results_methods[method] = {
        'mean_loss': np.mean(final_losses),
        'mean_ess': np.mean(final_ess)
    }
    
    print(f"  Final loss: {np.mean(final_losses):.3f}")
    print(f"  Final ESS: {np.mean(final_ess):.3f}")

# Cell 6: Visualize loss curves comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("FM Interpolant Comparison")
interp_vals = list(results_comparison.keys())
mean_losses = [results_comparison[k]['mean_loss'] for k in interp_vals]
std_losses = [results_comparison[k]['std_loss'] for k in interp_vals]
plt.errorbar(interp_vals, mean_losses, yerr=std_losses, marker='o')
plt.xlabel('FM Interpolant')
plt.ylabel('Final Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title("Method Comparison")
method_names = list(results_methods.keys())
method_losses = [results_methods[k]['mean_loss'] for k in method_names]
plt.bar(method_names, method_losses)
plt.ylabel('Final Loss')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Expected Results

**Will this succeed?** Based on the PA-FM implementation, you should expect:

1. **Successful Training**: The PA-FM loss should work with the existing flow models since it outputs a scalar loss compatible with PyTorch's autograd.

2. **Loss Components**: You'll see breakdown of:
   - Standard FM loss (when `fm_interpolant > 0`)
   - Augmented loss (from pseudo-targets)
   - ESS (Effective Sample Size) metrics showing diversity of supervision

3. **Performance Characteristics**:
   - **fm_interpolant=0.0**: Pure PA-FM, may be slower to converge initially but should provide better regularization
   - **fm_interpolant=1.0**: Pure standard FM, faster initial convergence
   - **fm_interpolant=0.5**: Balanced approach, likely best overall performance

4. **Key Metrics to Watch**:
   - `ess_combined`: Higher values (closer to batch_size) indicate more diverse supervision
   - `flow_alignment`: Should increase during training, indicating better alignment with true flow
   - `weights_augmented`: Shows how much the model learns from pseudo-targets

5. **Potential Challenges**:
   - The L2 method may work better for well-separated distributions
   - Temperature parameters (`pt_xt_y_tau`, `p_y_z_tau`) may need tuning for different datasets
   - Training might be slightly slower due to additional computations for pseudo-targets

The implementation should integrate smoothly with the existing notebook structure!
