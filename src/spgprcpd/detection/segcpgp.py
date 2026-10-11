"""SegCPGP: change-point detection with a gaussian process change-point kernel.

Fits a gaussian process whose kernel is a ChangePoints combination and tests the fitted change
location with a likelihood-ratio test against a single-kernel null, recursing into the
segments it splits off. Unlike the sequential detectors it is a retrospective method: it
sees the whole segment before deciding, which is why it is scored with
detect_only_change=True.

derived work. Adapted from https://github.com/JVerbeek/segcpgp.

Upstream licence: MIT.

modified from the original: adapted to this project's
spgprcpd.detection.base.CPD interface and its streaming alarms contract.
"""
import copy

import gpflow
import numpy as np
from gpflow.utilities import parameter_dict
from scipy.stats.distributions import chi2

from spgprcpd.detection.base import CPD
from spgprcpd.inference import sparse


class SEGCPGP(CPD):
    def __init__(self, dist_prior, kernel_init, enough_data=2, threshold = None, alpha_arl = None):
        super().__init__(threshold, alpha_arl, detect_only_change=True)
        self.dist_prior = dist_prior
        self.kernel_init = kernel_init
        self.models = {}
        self.best_model = None
        self.last_T = 0
        self.enough_data = enough_data
        self.LOCS = []
        self.STEEPNESS = []
        self.num_feature = None

    def get_all_params(self):
        return {
            "threshold": self.threshold,
        }

    def __get_high_likelihood_model(self, X, y, model_name, models, best_model, kernel_copy=1):
        def __new_copy_kernel(k):
            model = models["gpr"].model
            kernel = model.kernel

            kernel_cls = type(kernel)
            params = parameter_dict(kernel)
            param_values = {}
            for n, val in params.items():
                name = n.split(".")[-1]
                param_values[name] = val.numpy()
            new_kernel = kernel_cls(**param_values)

            if model_name == "cp":
                new_kernel = gpflow.kernels.ChangePoints(
                    [copy.deepcopy(new_kernel), copy.deepcopy(new_kernel)],
                    locations=[np.random.uniform(X[:,self.num_feature-1].min(), X[:,self.num_feature-1].max())],
                    steepness=[1]
                )
                new_kernel.active_dims = [self.num_feature-1]

            return new_kernel

        if model_name not in models:
            if model_name == "cp":
                k = copy.deepcopy(self.kernel_init)
                kernel = gpflow.kernels.ChangePoints(
                        [copy.deepcopy(k), copy.deepcopy(k)],
                        locations=[np.random.uniform(X[:,self.num_feature-1].min(), X[:,self.num_feature-1].max())],
                        steepness=[1.0],
                    )
                kernel.active_dims = [self.num_feature-1]
                model = sparse.Streaming_Sparse_GP(kernel=kernel)
            else:
                model = sparse.Streaming_Sparse_GP(kernel=copy.deepcopy(self.kernel_init))
        else:
            model = models["gpr"].copy()

        model.copy_kernel = __new_copy_kernel
        model.update(X, y)
        return model
    def _alarms(self, unselect_set, selected_set, values, l):
        alarms = []
        for v in values:
            if not (v[0] == -1 and v[1] == -1):
                alarms.append(v[1])
        return alarms

    def _values(self, unselect_set, selected_set, data, time_already):
        points = self.dist_prior(A = selected_set).points

        num_sensors = len(selected_set)
        time_steps = data.shape[0]
        begin_time = self.last_T

        if isinstance(self.dist_prior.kernel, np.ndarray):
            num_feature = self.dist_prior.kernel.shape[0]
        else:
            num_feature = len(self.dist_prior.kernel.lengthscales)

        if self.num_feature is None:
            self.num_feature = num_feature
        elif self.num_feature != num_feature:
            print("be carefull")


        y = np.full(data.shape[0] * data.shape[1], 0.0)
        X = np.full((time_steps*num_sensors, self.num_feature), 0.0)
        for i in range(time_steps):
            for j in range(num_sensors):
                if self.num_feature != 1:
                    X[i*num_sensors+j, :self.num_feature-1] = points[j]
                X[i*num_sensors+j, self.num_feature-1] = begin_time + i
            y[i*data.shape[1]:(i+1)*data.shape[1]] = data[i]

        t_raw = np.unique(X[:, self.num_feature-1])

        x_mean = np.nanmean(X, axis=0)
        x_std  = np.nanstd(X, axis=0)
        x_std[x_std == 0] = 1.0
        y_mean = np.nanmean(y)
        y_std  = np.nanstd(y)
        if y_std == 0:
            y_std = 1.0

        X = (X - x_mean) / x_std
        y = (y - y_mean) / y_std
        y = y.reshape(-1, 1)

        self.models, self.best_model = self.__compute(X, y, num_sensors)

        unique_times = np.unique(X[:, self.num_feature-1])

        cp_times = np.array(self.LOCS)

        closest_times = []
        closest_indices = []

        for cp in cp_times:
            idx = np.argmin(np.abs(unique_times - cp))
            closest_indices.append(idx)

        steep = self.STEEPNESS.copy()

        self.LOCS = []
        self.STEEPNESS = []


        values = []
        for idx in range(len(t_raw)):
            if idx in closest_indices:
                values.append((steep[closest_indices.index(idx)], int(t_raw[idx])))
            else:
                values.append((-1, -1))

        return values

    def __compute(self, X, y, num_sensors, models=None, best_model=None, kernel_copy=0):
        models = {} if models is None else models

        # Get single and changepoint model
        new_models = {}
        for model_name in ["cp", "gpr"]:
            model = self.__get_high_likelihood_model(
                X, y, model_name, models, best_model, kernel_copy=kernel_copy)
            new_models[model_name] = model

        cp = new_models["cp"]
        gpr = new_models["gpr"]
        location = cp.model.kernel.locations.numpy()
        location = location.item()
        steep = cp.model.kernel.steepness.numpy()
        steep = steep.item()

        # Compute LRT
        LRT = -2 * (gpr.model.maximum_log_likelihood_objective() -
                    cp.model.maximum_log_likelihood_objective())
        df = len(cp.model.trainable_parameters) - len(gpr.model.trainable_parameters)
        p = chi2.sf(LRT, df)

        # Try splitting
        # The null model is favored and we are done.
        if p > self.threshold or np.isnan(p):
            # The p-value is too large:
                # This means the detected change is not statistically significant
                # So the algorithm assumes: There is no reliable changepoint here
            # or the test failed or produced an invalid result:
                # Treated the same as no valid changepoint
            return new_models, "gpr"

        else:  # Split the signal -> there is a change point
            time_vals = np.unique(X[:, 0])

            # Check if t_0 is out of bounds
            if min(time_vals) > location or location > max(time_vals):
                return new_models, "cp"

            # Check if location not found, else return.
            if not np.isclose(location, self.LOCS, atol=1e-5).any():
                self.LOCS.append(location)
                self.STEEPNESS.append(steep)
            else:
                return models, "cp" # don't return new_models, because nothing have changed

            # Margin around changepoint to avoid detecting same changepoint multiple times
            try:
                epsilon = 5
                b2 = 5

                if location - epsilon <= time_vals.min(): # Edge case: if location is at the edge of the signal, do not use margin
                    b1 = 0
                if location + epsilon >= time_vals.max(): # Edge case: ""
                    b2 = 0

                split_left = list(map(int, time_vals)).index(int(location-epsilon))
                split_right = list(map(int, time_vals)).index(int(location+epsilon))

                X_left, X_right = X[:split_left*num_sensors], X[split_right*num_sensors:]
                y_left, y_right = y[:split_left*num_sensors], y[split_right*num_sensors:]

            except ValueError:
                split = list(map(int, time_vals)).index(int(location))
                X_left, X_right = X[:split*num_sensors], X[split*num_sensors:]
                y_left, y_right = y[:split*num_sensors], y[split*num_sensors:]

            # Recurse if signal is long enough.
            if len(X_left) > self.enough_data:
                # don't use the update model, because we look at the left data
                left_models = self.__compute(X_left, y_left, num_sensors, models, "cp", kernel_copy=0) # the copy the left kernel of the change point

            send_models = {} # don't send any model, recreate the model
            if len(X_right) > self.enough_data:
                # retrain the right model gpr, to use it for update
                new_models["gpr"] = self.__get_high_likelihood_model(
                    X_right, y_right, "gpr", {}, "")

                right_models = self.__compute(X_right, y_right, num_sensors, new_models, "cp", kernel_copy=1) # the copy the right kernel of the change point
                # save the right models, because we want to have the lastest model to the right
                send_models = right_models
            return send_models, "cp"
