import numpy as np

from spgprcpd.data.gaussian import create_dist_from_training_data


class Synthetic_Data_Manager:
    def __init__(
        self,
        num_samples,
        num_points,
        dist_prior,
        dist_post,
        change_time,
        hist_change_times=None,
        noise_std=0.1,
        N_test=0,
        num_features=3,
        grid=10,
        seed=0,
        seeds_point=None,
        random_walk=True,
        do_sample=True,
        do_estimation=False,
        extrapolation_cls=None,
        kernel_prior_init=None,
        kernel_post_init=None,
        est_time_output=False,
        only_testing_loc=True,
    ):
        self.num_samples = num_samples
        self.N_test = N_test
        self.num_points = num_points
        self.num_features = num_features
        self.dist_prior = dist_prior
        self._dist_prior = dist_prior
        self.dist_post = dist_post
        self._dist_post = dist_post   # preserved for None-check during estimation
        # change_time = tau3: the single change f0 -> f1 of the test (deployment) stream that
        #   the detectors run on and are scored against (cpd.eval(change_time), Exhaustive_CPD).
        # hist_change_times = (tau1, tau2): the recurring change f0 -> f1 -> f0 of the historical
        #   record used for estimation: f1 is observed in [tau1, tau2) so it can
        #   be estimated, then the process returns to f0.
        self.change_time = change_time
        self.hist_change_times = hist_change_times
        self.noise_std = noise_std    # sigma_eps: additive iid observation noise
        self.grid = grid
        self.seed = seed
        self.seeds_point = seeds_point
        self.random_walk = random_walk
        self.do_estimation = do_estimation
        self.extrapolation_cls = extrapolation_cls
        self.kernel_prior_init = kernel_prior_init
        self.kernel_post_init = kernel_post_init
        self.est_time_output = est_time_output
        self.only_testing_loc = only_testing_loc
        self.norm_saved = []
        self.test_change_time = (change_time - (hist_change_times[1] if hist_change_times is not None else 0)) if change_time is not None else change_time

        self.init(do_sample)

    def init(self, do_sample=True):
        if self.seeds_point is None:
            self.rnd = np.random.RandomState(self.seed)
            self.V = self.get_points()
        else:
            self.rnd = np.random.RandomState(self.seeds_point)
            self.V = self.get_points()
            self.rnd = np.random.RandomState(self.seed)

        self._dist_prior.update_points(self.V)
        if self._dist_post is not None:
            self._dist_post.update_points(self.V)
        self.X, self.y, self.y_iid = self.sample(do_sample)

        if self.do_estimation:
            self._build_distributions()

    # gP estimation of prior / post distributions

    def _build_flat_training_data(self, t_start, t_end):
        """Pack self.V and self.y_iid into (true negative, features+1) / (true negative, 1) arrays for gaussian process training.

        Rows are ordered time-major: all N sensors at t_start, then t_start+1, …, t_end-1.
        The last column of X is the absolute time index.
        """
        T, N = t_end - t_start, self.num_points
        X_flat = np.zeros((T * N, self.num_features + 1), dtype=np.float64)
        for t in range(T):
            rows = slice(t * N, (t + 1) * N)
            X_flat[rows, :-1] = self.V                       # spatial coords
            X_flat[rows, -1]  = float(t_start + t)           # absolute time
        y_flat = self.y[t_start:t_end].reshape(-1, 1).astype(np.float64)

        # X_flat is 2-D (T*N, features+1) by construction above, so normalise over the rows.
        # (There used to be a 3-D branch here guarded on an undefined name, `X_train`: dead
        # code that would have raised NameError had the shape ever differed.)
        x_mean = np.nanmean(X_flat, axis=0)
        x_std  = np.nanstd(X_flat, axis=0)
        x_std[x_std == 0] = 1.0

        y_mean = np.nanmean(y_flat)
        y_std  = np.nanstd(y_flat)
        if y_std == 0:
            y_std = 1.0

        X_flat_norm = (X_flat - x_mean) / x_std
        y_flat_norm  = (y_flat  - y_mean) / y_std

        return X_flat, y_flat, X_flat_norm, y_flat_norm

    def _build_distributions(self):
        """Estimate prior (and optionally post) distributions from sampled data.

        Fits a gaussian process on the iid increments for each regime, then wraps the
        trained kernel + training-data mean into a Multivariate_Gaussian.
        Call create_dist_from_model instead of create_dist_from_training_data
        if you need predictions at a custom set of spatial test points.
        """
        # Estimation reads the historical record (phase='historical'): f0 from its f0 block,
        # f1 from the observed [tau1, tau2) block of the recurring change f0 -> f1 -> f0.
        if self.hist_change_times is not None:
            tau1, tau2 = self.hist_change_times
            prior_range = slice(0, tau1)          # first f0 block (could pool [tau2:] under stationarity)
            post_range = slice(tau1, tau2)        # the observed f1 block
        else:
            prior_T = self.change_time if self.change_time is not None else self.num_samples
            prior_range = slice(0, prior_T)
            if self._dist_post is not None:
                post_range = slice(self.change_time, self.num_samples)
        all_loc = list(range(len(self.V)))

        # prior
        X_pr_norm, y_pr_norm, _, _, _, _  = self.get_dataset(all_loc, [], time_dimension=False, normalize=True, batch_size=None, time_range=prior_range, time_output=self.est_time_output, phase='historical')[0] # self._build_flat_training_data(0, prior_T)
        X_pr, y_pr, _, _, _, _ = self.get_dataset(all_loc, [], time_dimension=False, normalize=False, batch_size=None, time_range=prior_range, phase='historical')[0]
        gp_prior = self.extrapolation_cls(self.kernel_prior_init)
        gp_prior.update(X_pr_norm, y_pr_norm)

        self.dist_prior = create_dist_from_training_data(gp_prior, X_pr, y_pr, get_times=self.random_walk!=2)

        # post
        if self._dist_post is None:
            self.dist_post = None
        else:
            X_po_norm, y_po_norm, _, _, _, _  = self.get_dataset(all_loc, [], time_dimension=False, normalize=True, batch_size=None, time_range=post_range, time_output=self.est_time_output, phase='historical')[0]
            X_po, y_po, _, _, _, _ = self.get_dataset(all_loc, [], time_dimension=False, normalize=False, batch_size=None, time_range=post_range, phase='historical')[0]
            #X_po, y_po, X_po_norm, y_po_norm = self._build_flat_training_data(self.change_time, self.num_samples)
            gp_post = self.extrapolation_cls(self.kernel_post_init)
            gp_post.update(X_po_norm, y_po_norm)
            self.dist_post = create_dist_from_training_data(gp_post, X_po, y_po, get_times=self.random_walk!=2)

    def get_points(self):
        return self.rnd.uniform(0, self.grid, size=(self.num_points+self.N_test, self.num_features))

    def sample(self, do_sample=True):
        if self.random_walk == 1:
            return self.sample_random_walk(do_sample)
        elif self.random_walk == 0:
            return self.sample_uniform(do_sample)
        else:
            return self.sample_iid(do_sample)

    def _hist_is_post(self, i):
        """Whether historical time step i is in an f1 (post-change) regime.
        hist_change_times=(tau1,tau2) -> f0 -> f1 -> f0 (post only on [tau1, tau2)).        otherwise the legacy single change f0 -> f1 at change_time."""
        if self._dist_post is None:
            return False
        if self.change_time is None:
            return False
        if self.hist_change_times is not None:
            tau1, tau2 = self.hist_change_times
            return (tau1 <= i < tau2) or (i >= self.change_time)
        return i >= self.change_time

    def sample_uniform(self, do_sample=True):

        n = self.num_points + self.N_test

        X = np.zeros((self.num_samples, n, self.num_features)) # (time, sensor, xyz)
        y = np.zeros((self.num_samples, n))                    # (time, sensor)


        if not do_sample:
            return X, y, y

        def setup_dist(list_prior, list_post):
            self._dist_prior.update_points(self.V, list_prior)
            if self._dist_post is not None:
                if list_post is None:
                    self._dist_post.update_points(self.V)
                else:
                    self._dist_post.update_points(self.V, list_post)

        X[:] = self.V

        if self.hist_change_times is not None and self._dist_post is not None:
            # historical record f0 -> f1 -> f0
            tau1, tau2, ct = *self.hist_change_times, self.change_time
            prior_times = list(range(0, tau1)) + list(range(tau2, ct))
            post_times  = list(range(tau1, tau2)) + list(range(ct, self.num_samples))
            setup_dist(prior_times, post_times)
            yp = self._dist_prior.sample(self.rnd).reshape(len(prior_times), n)
            yq = self._dist_post.sample(self.rnd).reshape(len(post_times), n)
            for k, t in enumerate(prior_times): y[t] = yp[k]
            for k, t in enumerate(post_times):  y[t] = yq[k]
        elif self.change_time is None or self._dist_post is None:
            setup_dist(list(range(0, self.num_samples)), None)
            y = self._dist_prior.sample(self.rnd).reshape((self.num_samples - 0), n)
        else:                                          # legacy single change f0 -> f1
            setup_dist(list(range(0, self.change_time)), list(range(self.change_time, self.num_samples)))
            y[0:self.change_time] = self._dist_prior.sample(self.rnd).reshape((self.change_time - 0), n)
            y[self.change_time:self.num_samples] = self._dist_post.sample(self.rnd).reshape((self.num_samples - self.change_time), n)

        # additive iid observation noise eps ~ N(0, sigma_eps^2 I)
        if self.noise_std:
            y = y + self.noise_std * self.rnd.standard_normal(y.shape)

        return X, y, y

    def sample_random_walk(self, do_sample=True):
        self._dist_prior.update_points(self.V)

        if self._dist_post is not None:
            self._dist_post.update_points(self.V)


        X = np.zeros((self.num_samples, self.num_points+self.N_test, self.num_features)) # (time, sensor, xyz)
        y = np.zeros((self.num_samples, self.num_points+self.N_test))           # (time, sensor)
        y_iid  = np.zeros((self.num_samples, self.num_points+self.N_test))      # (time, sensor)

        if not do_sample:
            return X, y, y

        X[:] = self.V

        # historical record f0 -> f1 -> f0: increments drawn from the active
        # regime form a latent random walk, observed with additive iid noise eps ~ N(0, sigma_eps^2 I).
        # y is the observed level, y_iid its first difference (so y == cumsum(y_iid) still holds and
        # the CPD operates on the differenced series). With noise_std=0 this reduces to the old
        # behaviour (y_iid == the clean increments).
        n = self.num_points + self.N_test
        latent = np.zeros(n)
        prev_obs = np.zeros(n)
        for i in range(self.num_samples):
            d = self._dist_post if self._hist_is_post(i) else self._dist_prior
            latent = latent + d.sample(self.rnd)
            obs = latent
            if self.noise_std:
                obs = latent + self.noise_std * self.rnd.standard_normal(n)
            y[i] = obs
            y_iid[i] = obs - prev_obs
            prev_obs = obs
        return X, y, y_iid

    def sample_iid(self, do_sample=True):
        self._dist_prior.update_points(self.V)

        if self._dist_post is not None:
            self._dist_post.update_points(self.V)


        X = np.zeros((self.num_samples, self.num_points+self.N_test, self.num_features)) # (time, sensor, xyz)
        y = np.zeros((self.num_samples, self.num_points+self.N_test))           # (time, sensor)
        y_iid  = np.zeros((self.num_samples, self.num_points+self.N_test))      # (time, sensor)

        if not do_sample:
            return X, y, y

        X[:] = self.V

        # historical record f0 -> f1 -> f0: each step is an independent draw
        # from the active regime, observed with additive iid noise eps ~ N(0, sigma_eps^2 I).
        for i in range(self.num_samples):
            d = self._dist_post if self._hist_is_post(i) else self._dist_prior
            obs = d.sample(self.rnd)
            if self.noise_std:
                obs = obs + self.noise_std * self.rnd.standard_normal(obs.shape)
            y_iid[i] = obs
            y[i] = obs
        return X, y, y_iid

    def get_dataset(self, A, B, time_dimension=True, normalize=True, batch_size=None, time_range=slice(None), time_output=False, phase='test', save_norm=False):

        # Get full dataset first
        if self.only_testing_loc == 1:
            B = list(range(self.num_points, self.num_points+self.N_test))
        elif self.only_testing_loc == 2:
            B = B + list(range(self.num_points, self.num_points+self.N_test))

        if self.hist_change_times is not None and self._dist_post is not None:
            if phase == 'historical':
                if time_range == slice(None) or time_range is None:
                    time_range = slice(0, self.hist_change_times[1])
            elif phase == 'test':
                if time_range == slice(None) or time_range is None:
                    time_range = slice(self.hist_change_times[1], self.num_samples)



        # Sensor coordinates are static in time, so X comes from the sampled grid in both phases.
        X_train_full = self.X[time_range, A, :]
        X_test_full = self.X[time_range, B, :]

        y_train_full = self.y[time_range, A]
        y_test_full = self.y[time_range, B]
        y_iid_train_full = self.y_iid[time_range, A]
        y_iid_test_full = self.y_iid[time_range, B]

        complet_size = X_train_full.shape[0] # self.num_samples

        if batch_size is None:
            batch_size = complet_size

        # Split into batches along time dimension
        num_batches = int(np.ceil(complet_size / batch_size))

        resultats = []
        # Collect all batches without normalization
        all_X_train = []
        all_y_train = []
        all_y_iid_train = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, complet_size)

            X_train_batch = X_train_full[start_idx:end_idx, :, :]
            X_test_batch = X_test_full[start_idx:end_idx, :, :]
            y_train_batch = y_train_full[start_idx:end_idx, :]
            y_test_batch = y_test_full[start_idx:end_idx, :]
            y_iid_train_batch = y_iid_train_full[start_idx:end_idx, :]
            y_iid_test_batch = y_iid_test_full[start_idx:end_idx, :]

            if time_output:
                y_train_batch = y_train_batch.T
                y_test_batch = y_test_batch.T
                y_iid_train_batch = y_iid_train_batch.T
                y_iid_test_batch = y_iid_test_batch.T

                X_train_batch = X_train_batch[0, :, :]
                X_test_batch = X_test_batch[0, :, :]
            elif not time_dimension:
                time = np.arange(start_idx, end_idx).reshape(end_idx - start_idx, 1, 1)
                X_train_batch = np.concatenate(
                    [X_train_batch, np.repeat(time, len(A), axis=1)],
                    axis=2
                )
                X_train_batch = X_train_batch.reshape(-1, X_train_batch.shape[-1])
                X_test_batch = np.concatenate(
                    [X_test_batch, np.repeat(time, len(B), axis=1)],
                    axis=2
                )
                X_test_batch = X_test_batch.reshape(-1, X_test_batch.shape[-1])

                y_train_batch = y_train_batch.reshape(-1, 1)
                y_test_batch = y_test_batch.reshape(-1, 1)
                y_iid_train_batch = y_iid_train_batch.reshape(-1, 1)
                y_iid_test_batch = y_iid_test_batch.reshape(-1, 1)

                # Return only real measurements: drop the training (X, y) pairs whose target
                # is missing (nan). The gaussian process trains on observed points only. The test set is left
                # untouched so it stays aligned with the evaluation mask (inference is scored
                # on observed entries elsewhere).
                obs = ~np.isnan(np.asarray(y_train_batch, float)).reshape(-1)
                if not obs.all():
                    X_train_batch = X_train_batch[obs]
                    y_train_batch = y_train_batch[obs]
                    y_iid_train_batch = y_iid_train_batch[obs]

            # Append to lists (no normalization yet)
            all_X_train.append(X_train_batch)
            all_y_train.append(y_train_batch)
            all_y_iid_train.append(y_iid_train_batch)

            resultats.append((X_train_batch, y_train_batch, y_iid_train_batch, X_test_batch, y_test_batch, y_iid_test_batch))


        # Compute global normalization statistics
        if normalize:
            # Concatenate all batches
            X_train_all = np.concatenate(all_X_train, axis=0)
            # batches stack along time (axis 0) in the point-cloud layout and along the
            # sensor axis (axis 1) in the time-as-output layout. ValueError is what
            # numpy raises on the mismatch, so only that triggers the fallback
            try:
                y_train_all = np.concatenate(all_y_train, axis=0)
                y_iid_train_all = np.concatenate(all_y_iid_train, axis=0)
            except ValueError:
                y_train_all = np.concatenate(all_y_train, axis=1)
                y_iid_train_all = np.concatenate(all_y_iid_train, axis=1)

            if len(X_train_all.shape) == 2:
                x_mean = np.nanmean(X_train_all, axis=0)
                x_std = np.nanstd(X_train_all, axis=0)
            elif len(X_train_all.shape) == 3:
                x_mean = np.nanmean(X_train_all, axis=(0, 1))
                x_std = np.nanstd(X_train_all, axis=(0, 1))
            x_std[x_std == 0] = 1.0

            y_mean = np.nanmean(y_train_all)
            y_std = np.nanstd(y_train_all)
            if y_std == 0:
                y_std = 1.0

            y_iid_mean = np.nanmean(y_iid_train_all)
            y_iid_std = np.nanstd(y_iid_train_all)
            if y_iid_std == 0:
                y_iid_std = 1.0

            if save_norm:
                self.norm_saved = [[x_mean, x_std], [y_mean, y_std], [y_iid_mean, y_iid_std]]

            for i in range(len(resultats)):
                X_train_batch, y_train_batch, y_iid_train_batch, X_test_batch, y_test_batch, y_iid_test_batch = resultats[i]

                # Normalize the concatenated data
                X_train_batch = (X_train_batch - x_mean) / x_std
                X_test_batch = (X_test_batch - x_mean) / x_std
                y_train_batch = (y_train_batch - y_mean) / y_std
                y_test_batch = (y_test_batch - y_mean) / y_std
                y_iid_train_batch = (y_iid_train_batch - y_iid_mean) / y_iid_std
                y_iid_test_batch = (y_iid_test_batch - y_iid_mean) / y_iid_std

                resultats[i] = (X_train_batch, y_train_batch, y_iid_train_batch, X_test_batch, y_test_batch, y_iid_test_batch)

        return resultats


