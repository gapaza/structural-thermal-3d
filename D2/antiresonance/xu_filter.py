import numpy as np
from scipy.spatial import cKDTree


class XuProjectionFilter:
    def __init__(self, nelx, nely, filter_radius, elem_volume=1.0):
        """
        Initialize the filter with mesh dimensions and filter radius.
        Assumes a regular 2D grid for simplicity.
        """
        self.nelx = nelx
        self.nely = nely
        self.rmin = filter_radius
        self.elem_volume = elem_volume
        self.n_elem = nelx * nely

        # Precompute the weight matrix (H) and neighborhood indices
        self._precompute_weights()

    def _precompute_weights(self):
        """
        Precomputes neighborhood indices and weights for the density filter (Eq 8).
        Using cKDTree for efficient neighbor search on the grid.
        """
        # Generate element center coordinates (0.5, 1.5, ...)
        x = np.arange(self.nelx) + 0.5
        y = np.arange(self.nely) + 0.5
        xv, yv = np.meshgrid(x, y)  # Note: check your solver's flattening order (F vs C)
        centers = np.column_stack((xv.flatten(), yv.flatten()))

        # Build tree and query pairs within radius
        tree = cKDTree(centers)

        self.weights = []
        self.indices = []
        self.weight_sums = np.zeros(self.n_elem)

        # query_ball_point finds all points within distance r
        indices_list = tree.query_ball_point(centers, self.rmin)

        for e, neighbors in enumerate(indices_list):
            dists = np.linalg.norm(centers[neighbors] - centers[e], axis=1)

            # w_i = R_f - R_ie (linear cone filter)
            w = np.maximum(0, self.rmin - dists)

            # Incorporate element volume v_i into the weight (Eq 8 numerator)
            # v_i is constant here, but we include it for rigor.
            w = w * self.elem_volume

            self.weights.append(w)
            self.indices.append(neighbors)
            self.weight_sums[e] = np.sum(w)  # Denominator of Eq 8

    def density_filter(self, rho):
        """
        Applies the density filter (Eq 8).
        rho: flattened numpy array of design variables
        """
        rho_filtered = np.zeros_like(rho)

        for e in range(self.n_elem):
            # Numerator: sum(w_i * v_i * rho_i)
            # Denominator: sum(w_i * v_i) -> stored in self.weight_sums

            neighbor_indices = self.indices[e]
            w_vals = self.weights[e]

            numerator = np.dot(w_vals, rho[neighbor_indices])
            rho_filtered[e] = numerator / self.weight_sums[e]

        return rho_filtered

    def heaviside_projection(self, rho_bar, beta, eta):
        """
        Applies the specific Heaviside projection described in Eq 9.
        """
        # Prepare output array
        rho_tilde = np.zeros_like(rho_bar)

        # Mask for the two cases in Eq 9
        mask_low = (rho_bar >= 0) & (rho_bar <= eta)
        mask_high = (rho_bar > eta) & (rho_bar <= 1.0)

        # Case 1: 0 <= rho_bar <= eta
        if np.any(mask_low):
            r = rho_bar[mask_low]
            # Avoid division by zero if eta is exactly 0
            curr_eta = max(eta, 1e-10)

            term1 = np.exp(-beta * (1 - r / curr_eta))
            term2 = (1 - r / curr_eta) * np.exp(-beta)
            rho_tilde[mask_low] = curr_eta * (term1 - term2)

        # Case 2: eta < rho_bar <= 1
        if np.any(mask_high):
            r = rho_bar[mask_high]
            # Avoid division by zero if eta is exactly 1
            denom = max((1 - eta), 1e-10)

            ratio = (r - eta) / denom
            term1 = 1 - np.exp(-beta * ratio)
            term2 = ratio * np.exp(-beta)
            rho_tilde[mask_high] = denom * (term1 + term2) + eta

        return rho_tilde

    def apply_filter(self, rho, beta):
        """
        Full filter chain: Density Filter -> Bisection for Eta -> Heaviside Projection
        Returns: rho_projected, eta_optimal
        """
        # Step 1: Density Filter
        rho_bar = self.density_filter(rho)

        # Step 2: Bisection to find eta such that volume is preserved.
        # Target volume is the volume of the filtered density field.
        target_vol = np.sum(rho_bar)

        eta_min, eta_max = 0.0, 1.0
        eta = 0.5  # Initial guess

        # Bisection settings
        tol = 1e-5 * self.n_elem  # Tolerance
        max_iter = 50

        for _ in range(max_iter):
            eta = (eta_min + eta_max) / 2.0
            rho_tilde = self.heaviside_projection(rho_bar, beta, eta)
            new_vol = np.sum(rho_tilde)

            if abs(new_vol - target_vol) < tol:
                break

            # Logic: Increasing eta (threshold) generally DECREASES volume (removes material).
            # If our new_vol is too high, we need to increase eta.
            if new_vol > target_vol:
                eta_min = eta
            else:
                eta_max = eta

        return rho_tilde, eta


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Setup
    nelx, nely = 60, 30
    radius = 2.5
    beta = 5.0  # Continuation parameter (start small, increase during optimization)

    # 2. Instantiate Filter
    to_filter = XuProjectionFilter(nelx, nely, radius)

    # 3. Dummy Data (Random design variables)
    np.random.seed(42)
    rho_design = np.random.rand(nelx * nely)

    # 4. Execute Filter
    rho_projected, final_eta = to_filter.apply_filter(rho_design, beta)

    print(f"Original Volume:  {np.sum(rho_design):.4f}")
    print(f"Projected Volume: {np.sum(rho_projected):.4f}")
    print(f"Optimized Eta:    {final_eta:.4f}")

    # Validation
    if abs(np.sum(rho_projected) - np.sum(rho_design)) < 0.1:
        print("Success: Volume preserved.")
    else:
        print("Warning: Volume deviation detected.")