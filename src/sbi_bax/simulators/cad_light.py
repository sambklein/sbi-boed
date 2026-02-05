import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Shape Generator
# -----------------------------
def generate_shape(size=128, shape_type="bump", defect=False):
    """
    Generate a simple 2D heightmap representing a manufactured part.
    shape_type: "bump", "cylinder", "wave"
    defect: if True, add a local defect (dent/bump/scratch)
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    if shape_type == "bump":
        Z = 0.5 * np.exp(-((X**2 + Y**2) / 0.2))
    elif shape_type == "cylinder":
        Z = (X**2 + Y**2 < 0.5**2).astype(float) * 0.5
    elif shape_type == "wave":
        Z = 0.2 * np.sin(3 * X) * np.cos(3 * Y)
    else:
        Z = np.zeros_like(X)

    if defect:
        # Add a defect: small dent at random location
        cx, cy = np.random.uniform(-0.5, 0.5, 2)
        defect = -0.2 * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 0.01)
        Z += defect

    return Z


# -----------------------------
# Nuisance Sampler
# -----------------------------
def sample_nuisance():
    return {
        "albedo": np.random.uniform(0.6, 1.0),
        "ambient": np.random.uniform(0.0, 0.2),
        "gamma": np.random.normal(1.0, 0.1),
        "noise_std": np.random.uniform(0.01, 0.05),
        "illum_angle": np.random.uniform(-0.2, 0.2),  # projector tilt
        "calib_error": np.random.uniform(-0.05, 0.05),  # baseline shift
    }


# -----------------------------
# Forward Model (structured light)
# -----------------------------
def structured_light_forward(Z, nuisances, n_phase=4):
    size = Z.shape[0]
    x = np.linspace(0, 2 * np.pi, size)
    X, _ = np.meshgrid(x, x)

    patterns = []
    for k in range(n_phase):
        phase_shift = 2 * np.pi * k / n_phase
        pattern = nuisances["albedo"] * (
            0.5 + 0.5 * np.cos(X + nuisances["illum_angle"] * Z + phase_shift)
        )
        pattern = np.power(pattern + nuisances["ambient"], nuisances["gamma"])
        pattern += np.random.normal(0, nuisances["noise_std"], size=(size, size))
        patterns.append(pattern)

    return np.array(patterns)


# -----------------------------
# Example run
# -----------------------------
Z = generate_shape(size=128, shape_type="cylinder", defect=True)
nuis = sample_nuisance()
patterns = structured_light_forward(Z, nuis)

# Visualize
fig, axs = plt.subplots(1, patterns.shape[0] + 1, figsize=(15, 3))
axs[0].imshow(Z, cmap="viridis")
axs[0].set_title("Shape (heightmap)")
axs[0].axis("off")
for i, p in enumerate(patterns):
    axs[i + 1].imshow(p, cmap="gray")
    axs[i + 1].set_title(f"Phase {i}")
    axs[i + 1].axis("off")
plt.show()
