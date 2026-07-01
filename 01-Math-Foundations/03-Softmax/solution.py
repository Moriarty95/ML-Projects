import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        # Subtract the max for numerical stability
        z_stable = z - np.max(z)
        # Compute the exponentials
        exp_z = np.exp(z_stable)
        # Normalize to get probabilities
        softmax_z = exp_z / np.sum(exp_z)
        return np.round(softmax_z, 4)