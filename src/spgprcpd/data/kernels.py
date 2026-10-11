import gpflow
import numpy as np

np.random.seed(0)

kernels = {}

def registry(name, category, value_fn, time_series, func):
    if name not in kernels:
        kernels[name] = {}

    def factory():
        values = value_fn() if callable(value_fn) else value_fn
        if not isinstance(values, tuple):
            values = (values,)
        return func(*values)

    if time_series == 0:
        if "no_time" not in kernels[name]:
            kernels[name]["no_time"] = {}
        kernels[name]["no_time"][category] = factory
        if "time" not in kernels[name]:
            kernels[name]["time"] = {}
        kernels[name]["time"][category] = factory
    elif time_series == 1:
        if "no_time" not in kernels[name]:
            kernels[name]["no_time"] = {}
        kernels[name]["no_time"][category] = factory
    elif time_series == 2 :
        if "time" not in kernels[name]:
            kernels[name]["time"] = {}
        kernels[name]["time"][category] = factory
    elif time_series == 3 :
        if "learn_not_time" not in kernels[name]:
            kernels[name]["learn_not_time"] = {}
        kernels[name]["learn_not_time"][category] = factory

def RBF(lengthscale, variance):
    return gpflow.kernels.RBF(
        lengthscales=lengthscale,
        variance=variance
    )

def Matern32(lengthscale, variance):
    return gpflow.kernels.Matern32(
        lengthscales=lengthscale,
        variance=variance
    )

def Linear(variance):
    return gpflow.kernels.Linear(variance=variance)

def Periodic(lengthscale, variance, period):
    return gpflow.kernels.Periodic(
        base_kernel=gpflow.kernels.SquaredExponential(
            lengthscales=lengthscale,
            variance=variance
        ),
        period=period
    )

def Polynomial(degree, variance):
    return gpflow.kernels.Polynomial(
        degree=degree,
        variance=variance
    )

def Rational_Quadratic(variance, lengthscale, alpha):
    return gpflow.kernels.RationalQuadratic(
        variance=variance,
        lengthscales=lengthscale,
        alpha=alpha
    )

def Stationary(lengthscale, variance):
    return gpflow.kernels.Stationary(lengthscales=lengthscale, variance=variance)

def IsotropicStationary(lengthscale, variance):
    return gpflow.kernels.IsotropicStationary(lengthscales=lengthscale, variance=variance)

def AnisotropicStationary(lengthscale, variance):
    return gpflow.kernels.AnisotropicStationary(lengthscales=lengthscale, variance=variance)

registry("Matern_Anisotropic", "diff_3", lambda: ([11, 15], 1.0), 1, Matern32)
registry("Matern_Anisotropic", "diff_4", lambda: ([5, 7], 1.0), 1, Matern32)
registry("Matern_Anisotropic", "diff_5", lambda: ([5, 3], 1.0), 1, Matern32)
registry("Matern_Anisotropic", "random", lambda: ([np.random.rand()*4+8,np.random.rand()*4+8], 1.0), 1, Matern32)
registry("Matern_Anisotropic", "diff_3", lambda: ([11, 15], 1.0), 3, Matern32)
registry("Matern_Anisotropic", "diff_4", lambda: ([5, 7], 1.0), 3, Matern32)
registry("Matern_Anisotropic", "diff_5", lambda: ([5, 3], 1.0), 3, Matern32)
registry("Matern_Anisotropic", "random", lambda: ([np.random.rand()*4+8,np.random.rand()*4+8], 1.0), 3, Matern32)
registry("Matern_Anisotropic", "init_learning", lambda: ([0.8]*3, 1.0), 0, Matern32)
registry("Matern_Anisotropic", "init_learning", lambda: ([0.8]*2, 1.0), 3, Matern32)

registry("Matern_Anisotropic", "diff_3", lambda: ([11, 15, 9], 1.0), 2, Matern32)
registry("Matern_Anisotropic", "diff_4", lambda: ([5, 7, 8], 1.0), 2, Matern32)
registry("Matern_Anisotropic", "diff_5", lambda: ([5, 3, 6], 1.0), 2, Matern32)
registry("Matern_Anisotropic", "random", lambda: ([np.random.rand()*4+8,np.random.rand()*4+8,np.random.rand()*4+8], 1.0), 2, Matern32)
