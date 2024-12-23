import numpy as np

class Objective:
    def loss(self, y, pred): raise NotImplementedError
    def gradient(self, y, pred): raise NotImplementedError
    def hessian(self, y, pred): raise NotImplementedError


class SquaredErrorObjective(Objective):
    def loss(self, y, pred): return np.mean((y - pred) ** 2)
    def gradient(self, y, pred): return pred - y
    def hessian(self, y, pred): return np.ones(len(y))