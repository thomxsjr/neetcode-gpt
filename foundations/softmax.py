import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        output = []
        total = 0
        max_z = max(z)
        exp_i = []
        for i in z:
            score_i = np.exp(i - max_z)
            exp_i.append(score_i)
            total += score_i
        for j in exp_i:
            output.append(round(j/total, 4))
        return output
