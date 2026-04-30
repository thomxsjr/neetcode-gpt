import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        total_cost = 0
        n = len(y_true)
        for i in range(n):
            total_cost -=  (y_true[i]*np.log(y_pred[i])) + ((1-y_true[i])*np.log(1-y_pred[i]))
        loss = total_cost/n
        return round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        total_cost = 0
        n = len(y_true)
        c = len(y_true[0])
        for i in range(n):
            for j in range(c):
                total_cost -=  (y_true[i][j]*np.log(y_pred[i][j]))
        loss = total_cost/n
        return round(loss, 4)
