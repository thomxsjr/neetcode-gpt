class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        x = init
        for i in range(0, iterations):
            x = x - (learning_rate*2*x)
        if x == init:
            return x
        return round(x, 5)

