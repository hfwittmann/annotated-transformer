# %%
import matplotlib.pyplot as plt
import numpy as np
import torch


# %%
# The visualisation is from here: https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
def positional_encoding_visualization(d, L):
    pos = np.arange(L)[:, np.newaxis]
    i = np.arange(d)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d))
    angle_rads = pos * angle_rates
    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(angle_rads.T, cmap="viridis")
    plt.ylabel("Encoding Dimension")
    plt.xlabel("Position Index")
    plt.colorbar()
    plt.show()


# Example usage
positional_encoding_visualization(512, 100)


# %% [markdown]
# The visualisation below is (basically) from here:
# https://www.tensorflow.org/text/tutorials/transformer#positional_encoding

# %%
import numpy as np


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    sin_vals = np.sin(angle_rads)
    cos_vals = np.cos(angle_rads)

    pos_encoding = np.concatenate([sin_vals, cos_vals], axis=-1)

    return pos_encoding.astype(np.float32)


# %%
pos_encoding = positional_encoding(length=2048, depth=512)

# Check the shape.
print(pos_encoding.shape)

# Plot the dimensions.
plt.figure(figsize=(24, 16))
plt.pcolormesh(pos_encoding.T, cmap="RdBu")
plt.ylabel("Depth")
plt.xlabel("Position")
plt.colorbar()
plt.show()

import matplotlib.pyplot as plt

# %%
import numpy as np

# Replace tf.norm with np.linalg.norm
pos_encoding = positional_encoding(10000, 100)
pos_encoding /= np.linalg.norm(pos_encoding, axis=1, keepdims=True)
p = pos_encoding[1000]
dots = np.einsum("pd,d -> p", pos_encoding, p)
plt.subplot(2, 1, 1)
plt.plot(dots)
plt.ylim([0, 1])
plt.plot(
    [950, 950, float("nan"), 1050, 1050],
    [0, 1, float("nan"), 0, 1],
    color="k",
    label="Zoom",
)
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(dots)
plt.xlim([950, 1050])
plt.ylim([0, 1])
plt.show()


# %%


# %%


# %%
