import numpy as np
import matplotlib.pyplot as plt

# --- Vanilla Diffusion ---

vanilla_datapoints = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])

vanilla_comp_mean = np.array([10505, 6034, 1132, 10.1, 7.6, 0.32, 0.34, 0.12])
vanilla_comp_std = np.array([50062, 21653, 3920, 15.1, 38.4, 0.43, 0.65, 0.15])

vanilla_vfe_mean = np.array([0.079, 0.064, 0.025, 0.02, 0.012, 0.015, 0.011, 0.011])
vanilla_vfe_std = np.array([0.11, 0.078, 0.022, 0.01, 0.003, 0.007, 0.003, 0.003])


# --- Gradient Diffusion ---

grad_datapoints = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])

grad_comp_mean = np.array([504.4, 0.068, 0.042, 0.032, 0.030, 0.026, 0.025, 0.022])
grad_comp_std = np.array([2698.34, 0.023, 0.009, 0.008, 0.007, 0.005, 0.005, 0.004])

grad_vfe_mean = np.array([0.043, 0.023, 0.016, 0.013, 0.012, 0.007, 0.008, 0.006])
grad_vfe_std = np.array([0.034, 0.018, 0.012, 0.004, 0.006, 0.003, 0.003, 0.003])


# --- Gradient Augmented Diffusion ---

grad_aug_datapoints = np.array([100, 200, 300, 400, 500, 1000, 1500])

grad_aug_comp_mean = np.array([0.15, 0.128, 0.065, 0.056, 0.054, 0.038, 0.022])
grad_aug_comp_std = np.array([0.05, 0.06, 0.02, 0.003, 0.016, 0.019, 0.008])

grad_aug_vfe_mean = np.array([0.05, 0.04, 0.023, 0.012, 0.008, 0.007, 0.004])
grad_aug_vfe_std = np.array([0.015, 0.012, 0.007, 0.004, 0.004, 0.0024, 0.0015])


# --- Logic to Clip Error Bars ---
def get_clipped_yerr(mean, std, floor=1e-2):
    """
    Calculates asymmetric errors so the bottom doesn't drop below 'floor'.
    """
    # Lower error is the distance from the mean to our desired bottom limit
    # We want the bottom to be: max(mean - std, floor)
    # The 'yerr' lower value is: mean - bottom
    lower_err = mean - np.maximum(mean - std, floor)
    lower_err = np.maximum(lower_err, 0.0)

    # Upper error remains the standard deviation
    upper_err = std

    return [lower_err, upper_err]


# Generate the asymmetric error arrays
vanilla_comp_yerr = get_clipped_yerr(vanilla_comp_mean, vanilla_comp_std)
grad_comp_yerr = get_clipped_yerr(grad_comp_mean, grad_comp_std)
grad_aug_comp_yerr = get_clipped_yerr(grad_aug_comp_mean, grad_aug_comp_std)

vanilla_vfe_yerr = get_clipped_yerr(vanilla_vfe_mean, vanilla_vfe_std)
grad_vfe_yerr = get_clipped_yerr(grad_vfe_mean, grad_vfe_std)
grad_aug_vfe_yerr = get_clipped_yerr(grad_aug_vfe_mean, grad_aug_vfe_std)


# -------------- Compliance Plotting --------------
plt.figure(figsize=(10, 6))

plt.errorbar(vanilla_datapoints, vanilla_comp_mean, yerr=vanilla_comp_yerr,
             label='Vanilla Diffusion', color='orange', marker='o',
             linestyle='-', capsize=5)

plt.errorbar(grad_datapoints, grad_comp_mean, yerr=grad_comp_yerr,
             label='Gradient Diffusion', color='blue', marker='s',
             linestyle='-', capsize=5)

plt.errorbar(grad_aug_datapoints, grad_aug_comp_mean, yerr=grad_aug_comp_yerr,
                label='Gradient Diffusion (Noised)', color='green', marker='^',
                linestyle='-', capsize=5)



plt.yscale('log')
plt.xlabel('Dataset Size', fontsize=14)
plt.ylabel('Structural Compliance (percent degradation)', fontsize=14)
plt.title('Structural Compliance vs. Dataset Size', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()
plt.show()



# -------------- Volume Fraction Error Plotting --------------
plt.figure(figsize=(10, 6))

plt.errorbar(vanilla_datapoints, vanilla_vfe_mean, yerr=vanilla_vfe_yerr,
             label='Vanilla Diffusion', color='orange', marker='o',
             linestyle='-', capsize=5)

plt.errorbar(grad_datapoints, grad_vfe_mean, yerr=grad_vfe_std,
                label='Gradient Diffusion', color='blue', marker='s',
                linestyle='-', capsize=5)

plt.errorbar(grad_aug_datapoints, grad_aug_vfe_mean, yerr=grad_aug_vfe_std,
                label='Gradient Diffusion (Noised)', color='green', marker='^',
                linestyle='-', capsize=5)

# plt.yscale('log')
plt.xlabel('Dataset Size', fontsize=14)
plt.ylabel('Volume Fraction Error', fontsize=14)
plt.title('Volume Fraction Error vs. Dataset Size', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()
plt.show()











