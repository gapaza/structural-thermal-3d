import numpy as np
import matplotlib.pyplot as plt

# --- Vanilla Diffusion ---

vanilla_datapoints = np.array([50, 100, 200, 300, 400, 500, 600, 700, 1000])

vanilla_comp_mean = np.array([3.9, 3.82, 0.51, 0.31, 0.21, 0.11, 0.08, 0.07, 0.04])
vanilla_comp_std = np.array([8.29, 10.76, 0.28, 0.16, 0.08, 0.04, 0.034, 0.03, 0.01])

vanilla_vfe_mean = np.array([0.12, 0.14, 0.093, 0.045, 0.023, 0.015, 0.011, 0.008, 0.008])
vanilla_vfe_std = np.array([0.13, 0.147, 0.09, 0.065, 0.033, 0.009, 0.006, 0.007, 0.004])


# --- Gradient Diffusion ---

grad_datapoints = np.array([50, 100, 200, 300, 400, 500])

grad_comp_mean = np.array([0.12, 0.11, 0.038, 0.027, 0.024, 0.023])
grad_comp_std = np.array([0.032, 0.038, 0.012, 0.008, 0.01, 0.007])

grad_vfe_mean = np.array([0.038, 0.03, 0.008, 0.005, 0.0047, 0.0043])
grad_vfe_std = np.array([0.017, 0.02, 0.004, 0.003, 0.002, 0.001])


# --- Gradient Augmented Diffusion ---

grad_aug_datapoints = np.array([50, 100, 200, 300])

grad_aug_comp_mean = np.array([0.09, 0.05, 0.021, 0.017])
grad_aug_comp_std = np.array([0.018, 0.012, 0.007, 0.005])

grad_aug_vfe_mean = np.array([0.02, 0.01, 0.004, 0.003])
grad_aug_vfe_std = np.array([0.006, 0.005, 0.002, 0.001])


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
plt.ylabel('Thermal Compliance (percent degradation)', fontsize=14)
plt.title('Thermal Compliance vs. Dataset Size', fontsize=16)
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











