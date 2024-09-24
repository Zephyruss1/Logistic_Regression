import numpy as np
import cvxpy as cp
import inspect
import torch
from scipy.optimize import minimize

EPSILON = 1e-5

class LogisticRegression:
    def __init__(self, args, X_train, Y_train, X_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

        self.num_samples = self.X_train.shape[0]
        self.dimension = self.X_train.shape[1]

        self._weights = np.zeros_like(X_train[0])
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.iter = args.iteration
        self.gamma = args.gamma

        print("============= CVX solving =============")
        self.opt_weights, self.opt_obj = self.CVXsolve()
        print("============= CVX solved =============")

        self.pre_Hessian = self.Hessian(self.weights)

        # Initialize AdamW and SGDW specific variables
        self.m = np.zeros_like(self.weights)
        self.v = np.zeros_like(self.weights)
        self.t = 0

    # UPDATE: Added property decorator for weights.
    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        if isinstance(value, np.ndarray) and value.shape == self._weights.shape:
            self._weights = value
        else:
            raise ValueError("Invalid value for weights.")

    # sigmoid function
    @staticmethod
    def sigmoid(input):
        input_clipped = np.clip(input, -500, 500) # Avoid overflow
        assert input_clipped.shape == input.shape, (
            f"input_clipped.shape: {input_clipped.shape}, "
            f"input.shape: {input.shape}"
        )
        return 1. / (1. + np.exp(-input_clipped))

    def getTest(self):
        self.test = self.sigmoid(self.X_test @ self.weights)
        return self.test

    def objective(self, weights):
        """
        return the objective value of the problem
        note that the objective is averaged over all samples
        """
        sigWeights = self.sigmoid(self.X_train @ weights)
        matGrad = self.Y_train * np.log(sigWeights + EPSILON)\
             + (1.0 - self.Y_train) * np.log(1 - sigWeights + EPSILON)
        return - np.sum(matGrad) / self.num_samples + 0.5 *\
            self.gamma * np.linalg.norm(weights) ** 2

    def gradient(self, weights):
        """
        return the gradient of objective function
        note that the gradient is averaged over all samples
        """
        sigWeights = self.sigmoid(self.X_train @ weights)
        matGrad = self.X_train.T @ (sigWeights - self.Y_train)
        return matGrad / self.num_samples + self.gamma * weights

    def Hessian(self, weights):
        """
        return the Hessian of objective function
        note that the Hessian is averaged over all samples
        """
        sigWeights = self.sigmoid(self.X_train @ weights)
        D_diag = np.diag(sigWeights * (1 - sigWeights))
        return self.X_train.T @ D_diag @ self.X_train / self.num_samples\
            + self.gamma * np.identity(self.dimension)

    def update(self):
        # Optimization: Using mapping
        optimizer_methods = {
            "GD"                    : self.gradient(self.weights),
            "ModifiedNewton"        : self.modified_newton,
            "ModifiedNewtonArmijo"  : self.modified_newton_with_armijo,
            "ConjugateGradient"     : self.conjugate_gradient,
            "ConjugateGDArmijo"     : self.conjugate_gradient_with_armijo,
            "LevenbergMarquardt"    : self.levenberg_marquardt,
            "BFGS"                  : self.BFGS,
            "LBFGS"                 : self.LBFGS,
            "GDArmijo"              : self.gradient_descent_with_armijo,
            "Adam"                  : self.adam,
            "AdamW"                 : self.adamw,
            "SGD"                   : self.sgd,
            "SGDW"                  : self.sgdw,
            "NelderMead"            : self.nelder_mead
        }
        if self.optimizer == 'GD':
            gradient = self.gradient(self.weights)
            update_direction = gradient
            self.weights -= self.lr * update_direction
        elif self.optimizer in optimizer_methods:
            return optimizer_methods[self.optimizer]()
        else:
            raise NotImplementedError
        a, b = self.diff_cal(self.weights)
        return a, b

    def modified_newton(self):
        gradient = self.gradient(self.weights)
        hessian = self.Hessian(self.weights)

        update_direction = np.linalg.inv(hessian) @ gradient

        self.weights -= self.lr * update_direction

        a, b = self.diff_cal(self.weights)
        return a, b

    def conjugate_gradient(self):
        gradient = self.gradient(self.weights)

        if not hasattr(self, 'last_update'):
            self.last_update = gradient
            update_direction = -gradient
        else:
            beta = np.dot(gradient, gradient) / np.dot(self.last_update,
                                                        self.last_update)
            update_direction = -gradient + beta * self.last_update
            self.last_update = gradient

        self.weights -= self.lr * update_direction

        a, b = self.diff_cal(self.weights)
        return a, b

    def levenberg_marquardt(self):
        gradient = self.gradient(self.weights)
        hessian = self.Hessian(self.weights)

        damping_term = self.gamma * np.identity(self.dimension)

        update_direction = np.linalg.inv(hessian + damping_term) @ gradient

        self.weights -= self.lr * update_direction

        a, b = self.diff_cal(self.weights)
        return a, b

    def BFGS(self):
        gradient = self.gradient(self.weights)

        if not hasattr(self, 'B'):
            self.B = np.eye(self.dimension)
            self.last_update = np.zeros_like(gradient)
            self.last_gradient = np.zeros_like(gradient)

        y = gradient - self.last_gradient
        s = self.weights - self.last_update

        rho = 1 / np.dot(y, s) if np.dot(y, s) != 0 else 1000
        V = np.eye(self.dimension) - rho * np.outer(y, s)

        self.B = V.T @ self.B @ V + rho * np.outer(s, s)

        update_direction = -self.B @ gradient

        self.weights -= self.lr * update_direction

        self.last_update = self.weights
        self.last_gradient = gradient

        a, b = self.diff_cal(self.weights)
        return a, b

    # -------------------DEBUG MODE----------------------- #
    def LBFGS(self):
        if not isinstance(self.weights, torch.Tensor):
            self.weights = 0.1 * np.random.randn(self.dimension)

        initial_weights = self.weights.detach().numpy() if isinstance\
        (self.weights, torch.Tensor) else self.weights
        lbfgs = minimize(self.objective, initial_weights,\
            method='L-BFGS-B')
        self.weights = torch.tensor(lbfgs.x, requires_grad=True,\
            dtype=torch.float32)
        a, b = self.diff_cal(self.weights.detach().numpy())
        return a, b
    # -------------------DEBUG MODE----------------------- #
    def nelder_mead(self):
        res = minimize(self.objective, self.weights, method='nelder-mead',
                        options={'maxiter': 1000, 'xatol': 1e-8,
                                  'fatol': 1e-8, 'disp': False})
        self.weights = res.x
        a, b = self.diff_cal(self.weights)
        return a, b

    def gradient_descent_with_armijo(self):
        gradient = self.gradient(self.weights)
        direction = -gradient
        step_size = self.armijo_stepsize_search(gradient, self.weights,\
                                                direction)
        self.weights += step_size * direction
        a, b = self.diff_cal(self.weights)
        return a, b

    def modified_newton_with_armijo(self):
        gradient = self.gradient(self.weights)
        hessian = self.Hessian(self.weights)
        direction = np.linalg.solve(hessian, gradient)
        step_size = self.armijo_stepsize_search(gradient, self.weights,\
            direction)
        self.weights += step_size * direction
        a, b = self.diff_cal(self.weights)
        return a, b

    def conjugate_gradient_with_armijo(self):
        gradient = self.gradient(self.weights)
        direction = -gradient
        step_size = self.armijo_stepsize_search(gradient, self.weights,\
            direction)
        self.weights += step_size * direction
        a, b = self.diff_cal(self.weights)
        return a, b

    def armijo_stepsize_search(self, gradient, current_point, direction,
                            alpha=1.0, beta=0.5, max_iter=100, c=0.0001):
        t = alpha
        iter_count = 0
        while iter_count < max_iter:
            new_point = current_point + t * direction
            new_value = self.objective(new_point)
            if new_value <= self.objective(current_point) + c * t *\
                np.dot(gradient, direction):
                return t
            else:
                t *= beta
                iter_count += 1
        return t

    def CVXsolve(self):
        """
        use CVXPY to solve optimal solution
        """
        x = cp.Variable(self.dimension)
        objective = cp.sum(cp.multiply(self.Y_train, self.X_train @ x)\
                        - cp.logistic(self.X_train @ x))
        prob = cp.Problem(cp.Maximize(objective / self.num_samples - 0.5\
                                    * self.gamma * cp.norm2(x) ** 2))
        prob.solve(solver=cp.ECOS_BB, verbose=False)  # False if not print it

        opt_weights = np.array(x.value)
        opt_obj = self.objective(opt_weights)
        return opt_weights, opt_obj

    def diff_cal(self, weights):
        """
        Calculate the difference of input model weights with optimal in terms of:
        * 'weights'
        * 'objective'
        """
        weight_diff = np.linalg.norm(weights - self.opt_weights)
        obj_diff = abs(self.objective(weights) - self.opt_obj)
        return weight_diff, obj_diff

    # UPDATE: Adam optimizer
    def adam(self, beta1=0.9, beta2=0.999, EPSILON=1e-8):
        self.t += 1
        gradient = self.gradient(self.weights)
        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * (gradient ** 2)

        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)

        self.weights -= self.lr * m_hat / (np.sqrt(v_hat) + EPSILON)

        a, b = self.diff_cal(self.weights)
        return a, b

    # UPDATE: AdamW optimizer
    def adamw(self, beta1=0.9, beta2=0.999, EPSILON=1e-8, weight_decay=0.01,
            device='cpu'):
        if torch.cuda.is_available() and torch.version.hip is not None:
            device = 'cuda'
        
        self.t += 1
        gradient = self.gradient(self.weights)
        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * (gradient ** 2)

        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)

        self.weights -= self.lr * m_hat / (np.sqrt(v_hat) + EPSILON)\
            + weight_decay * self.weights

        a, b = self.diff_cal(self.weights)
        fused_aviable = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_aviable and device == 'cuda'
        print(f"using fused adam: {use_fused}")
        return a, b

    # UPDATE: SGD optimizer
    def sgd(self):
        gradient = self.gradient(self.weights)
        self.weights -= self.lr * gradient

        a, b = self.diff_cal(self.weights)
        return a, b

    # UPDATE: SGDW optimizer
    def sgdw(self, weight_decay=0.01):
        gradient = self.gradient(self.weights)
        self.weights -= self.lr * gradient + weight_decay * self.weights

        a, b = self.diff_cal(self.weights)
        return a, b
