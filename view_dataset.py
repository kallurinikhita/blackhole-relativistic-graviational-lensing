import torch
import matplotlib.pyplot as plt

import torch
import matplotlib.pyplot as plt

data = torch.load("dataset.pt", weights_only=False)

X = data["X"]
Y = data["Y"]

print("X shape:", X.shape)
print("Y shape:", Y.shape)

plt.imshow(X[0], cmap="inferno")
plt.title("Sigma")
plt.colorbar()
plt.show()


plt.figure()
plt.imshow(Y[0], cmap="coolwarm")
plt.title("Deflection X")
plt.colorbar()

plt.show()