import h5py
import numpy as np
import torch

f = h5py.File("snapshot.hdf5", "r")

pos = f["PartType0/Coordinates"][:]
mass = f["PartType0/Masses"][:]
vel = f["PartType0/Velocities"][:]

# create surface density- Σ(x, y) = Σ m_i * kernel(distance)
from scipy.stats import binned_statistic_2d

Sigma, xedges, yedges, _ = binned_statistic_2d(
    pos[:,0],
    pos[:,1],
    mass,
    statistic="sum",
    bins=256
)

# Compute gravitational potential proxy
# approximate Φ(x) ≈ convolution(Σ, 1/r)
from scipy.signal import fftconvolve

kernel = 1 / (np.sqrt(X**2 + Y**2) + 1e-3)
Phi = fftconvolve(Sigma, kernel, mode="same")

# Compute lensing field- deflection field: Δu = ∇Φ
# defines the gravitational lensing deflection angle as the gradient of a two-dimensional scalar potential
# describes how a massive body (the lens) bends light from a distant source, causing its image to appear shifted, distorted, or magnified
dPhi_dx, dPhi_dy = np.gradient(Phi)

deflection_x = dPhi_dx
deflection_y = dPhi_dy


# ML training pairs
X = [
    Sigma,     # density map
    Phi        # potential map
]

Y = [
    deflection_x,
    deflection_y
]

# normalize data
X = (X - X.mean()) / X.std()
Y = (Y - Y.mean()) / Y.std()

torch.save(dataset, "dataset.pt")