import h5py
import numpy as np
import torch
from scipy.stats import binned_statistic_2d
from scipy.signal import fftconvolve

#IllustrisTNG dataset

# INPUT (X):
# - matter distribution (Σ)
# - gravitational field (Φ)
# X = “what space looks like”
# (gravity + mass)

# OUTPUT (Y):
# - how spacetime bends light rays (Δu)
# Y = “how light bends through it”
# How does an object warp images around it

# realistic simulation data
pos = np.random.rand(1000, 3)
vel = np.random.rand(1000, 3)
mass = np.random.rand(1000)

with h5py.File("snapshot.hdf5", "w") as f:

    pt0 = f.create_group("PartType0")

    pt0.create_dataset("Coordinates", data=pos)
    pt0.create_dataset("Velocities", data=vel)
    pt0.create_dataset("Masses", data=mass)

    header = f.create_group("Header")
    header.attrs["Redshift"] = 0.5
    header.attrs["BoxSize"] = 100.0

# STEP 2 — READ + BUILD DATASET
with h5py.File("snapshot.hdf5", "r") as f:
    pos = f["PartType0/Coordinates"][:]
    mass = f["PartType0/Masses"][:]

# STEP 3 — Density projection
Sigma = np.zeros((256, 256))

for i in range(len(pos)):
    x, y = pos[i, 0], pos[i, 1]
    m = mass[i]

    # convert to grid index
    ix = int(x * 255)
    iy = int(y * 255)

    # spread mass in small neighborhood
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if 0 <= ix+dx < 256 and 0 <= iy+dy < 256:
                weight = np.exp(-(dx**2 + dy**2)/2)
                Sigma[ix+dx, iy+dy] += m * weight

# STEP 4 — Potential field
x = np.linspace(-1, 1, 256)
y = np.linspace(-1, 1, 256)
X, Y = np.meshgrid(x, y)

kernel = 1 / (np.sqrt(X**2 + Y**2) + 1e-3)
Phi = fftconvolve(Sigma, kernel, mode="same")

# STEP 5 — Lensing field
dPhi_dx, dPhi_dy = np.gradient(Phi)

deflection_x = dPhi_dx
deflection_y = dPhi_dy

# STEP 6 — ML dataset
X_data = np.stack([Sigma, Phi], axis=0)
Y_data = np.stack([deflection_x, deflection_y], axis=0)

# STEP 7 — Normalize
X_data = (X_data - X_data.mean()) / (X_data.std() + 1e-8)
Y_data = (Y_data - Y_data.mean()) / (Y_data.std() + 1e-8)

# STEP 8 — Save dataset
torch.save({
    "X": X_data,
    "Y": Y_data
}, "dataset.pt")

