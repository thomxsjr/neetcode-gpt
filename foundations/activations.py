import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        output = []
        for i in z:
            sig = 1/(1+np.exp(-i))
            output.append(round(sig, 5))
        return output

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        output = []
        for i in z:
            if i > 0:
                output.append(i)
            else:
                output.append(0.0)
        return output
