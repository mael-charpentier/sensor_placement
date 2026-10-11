"""sparse gaussian process models

derived work. Adapted from https://github.com/thangbui/streaming_sparse_gp, which
implements Bui, Yan & Turner, "A unifying framework for Gaussian process pseudo-point
approximations using Power Expectation Propagation" (JMLR 2017) and the streaming
formulation of Bui, Nguyen & Turner (2017).

Upstream licence: Apache-2.0.

modified from the original, as Apache-2.0 section 4(b) requires this notice to state:
the models were ported to the GPflow 2 API and reshaped to fit this project's
spgprcpd.inference.base.Extrapolation interface.
"""
import gpflow
import numpy as np
import tensorflow as tf
from gpflow import covariances
from gpflow.inducing_variables import InducingPoints
from gpflow.models import GPModel, InternalDataTrainingLossMixin


class SGPR_PEP(GPModel, InternalDataTrainingLossMixin):

    def __init__(self, data, kernel, Z, alpha, mean_function=None):
        self.X, self.Y = self.data = gpflow.models.util.data_input_to_tensor(data)
        likelihood = gpflow.likelihoods.Gaussian()
        num_latent_gps = GPModel.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        self.inducing_variable = InducingPoints(Z)
        self.num_data = self.X.shape[0]

        self.alpha = alpha

    def _common_terms(self):
        Mb = self.inducing_variable.num_inducing
        # configurable through gpflow's default jitter so a failed fit can be retried with
        # a larger value (Extrapolation._optimize_with_jitter). Never below the 1e-4 the
        # power ep models were tuned with
        jitter = gpflow.utilities.to_default_float(max(gpflow.config.default_jitter(), 1e-4))
        sigma2 = self.likelihood.variance
        alpha = self.alpha

        # b is inducing points
        # f is training points
        Kfdiag = self.kernel(self.X, full_cov=False)
        Kbf = covariances.Kuf(self.inducing_variable, self.kernel, self.X)
        Kbb = covariances.Kuu(self.inducing_variable, self.kernel, jitter=jitter)

        err = self.Y - self.mean_function(self.X)
        Lb = tf.linalg.cholesky(Kbb)
        Lbinv_Kbf = tf.linalg.triangular_solve(Lb, Kbf, lower=True)

        Qff_diag = tf.reduce_sum(tf.square(Lbinv_Kbf), axis=0)
        Dff = sigma2 + alpha * (Kfdiag - Qff_diag)
        Lbinv_Kbf_LDff = Lbinv_Kbf / tf.sqrt(Dff)
        d1 = tf.matmul(Lbinv_Kbf_LDff, Lbinv_Kbf_LDff, transpose_b=True)
        D = tf.eye(Mb, dtype=d1.dtype) + d1
        LD = tf.linalg.cholesky(D)

        Sinv_y = self.Y / tf.reshape(Dff, [self.num_data, 1])
        c = tf.matmul(Lbinv_Kbf, Sinv_y)
        LDinv_c = tf.linalg.triangular_solve(LD, c, lower=True)

        return (Kbf, Kbb, Lb, D, LD, LDinv_c, err, Dff)

    def maximum_log_likelihood_objective(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """

        sigma2 = self.likelihood.variance
        N = self.num_data
        alpha = self.alpha

        # a is old inducing points, b is new
        # f is training points
        (Kbf, Kbb, Lb, D, LD, LDinv_c, err, Dff) = self._common_terms()

        # constant term
        bound = -0.5 * N * np.log(2 * np.pi)
        # quadratic term
        bound += -0.5 * tf.reduce_sum(tf.square(err) / tf.reshape(Dff, [N, 1]))
        bound += 0.5 * tf.reduce_sum(tf.square(LDinv_c))

        # log det term
        bound += -0.5 * tf.reduce_sum(tf.math.log(Dff))
        bound += - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LD)))

        # trace-like term
        bound += - 0.5 * (1 - alpha) / alpha * \
            tf.reduce_sum(tf.math.log(Dff / sigma2))

        return bound

    def predict_f(self, Xnew, full_cov=False):
        # configurable through gpflow's default jitter so a failed fit can be retried with
        # a larger value (Extrapolation._optimize_with_jitter). Never below the 1e-4 the
        # power ep models were tuned with
        jitter = gpflow.utilities.to_default_float(max(gpflow.config.default_jitter(), 1e-4))

        # b is inducing points
        # f is training points
        # s is test points
        Kbs = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)
        (Kbf, Kbb, Lb, D, LD, LDinv_c, err, Dff) = self._common_terms()

        Lbinv_Kbs = tf.linalg.triangular_solve(Lb, Kbs, lower=True)
        LDinv_Lbinv_Kbs = tf.linalg.triangular_solve(LD, Lbinv_Kbs, lower=True)
        mean = tf.matmul(LDinv_Lbinv_Kbs, LDinv_c, transpose_a=True)

        if full_cov:
            Kss = self.kernel(Xnew) + jitter * tf.eye(tf.shape(Xnew)[0], dtype=gpflow.default_float())
            var1 = Kss
            var2 = - tf.matmul(Lbinv_Kbs, Lbinv_Kbs, transpose_a=True)
            var3 = tf.matmul(LDinv_Lbinv_Kbs, LDinv_Lbinv_Kbs, transpose_a=True)
            var = var1 + var2 + var3
        else:
            var1 = self.kernel(Xnew, full_cov=False) + jitter
            var2 = -tf.reduce_sum(tf.square(Lbinv_Kbs), axis=0)
            var3 = tf.reduce_sum(tf.square(LDinv_Lbinv_Kbs), axis=0)
            var = var1 + var2 + var3

        return mean + self.mean_function(Xnew), var

class OSGPR_PEP(GPModel, InternalDataTrainingLossMixin):
    def __init__(self, data, kernel, mu_old, Su_old, Kaa_old, Z_old, Z, alpha,
            mean_function=None):
        self.X, self.Y = self.data = gpflow.models.util.data_input_to_tensor(data)
        likelihood = gpflow.likelihoods.Gaussian()
        num_latent_gps = GPModel.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        self.inducing_variable = InducingPoints(Z)
        self.num_data = self.X.shape[0]
        self.alpha = alpha

        self.mu_old = tf.Variable(mu_old, shape=tf.TensorShape(None), trainable=False)
        self.M_old = Z_old.shape[0]
        self.Su_old = tf.Variable(Su_old, shape=tf.TensorShape(None), trainable=False)
        self.Kaa_old = tf.Variable(Kaa_old, shape=tf.TensorShape(None), trainable=False)
        self.Z_old = tf.Variable(Z_old, shape=tf.TensorShape(None), trainable=False)

    def _common_terms(self):
        Mb = self.inducing_variable.num_inducing
        Ma = self.M_old
        # configurable through gpflow's default jitter so a failed fit can be retried with
        # a larger value (Extrapolation._optimize_with_jitter). Never below the 1e-4 the
        # power ep models were tuned with
        jitter = gpflow.utilities.to_default_float(max(gpflow.config.default_jitter(), 1e-4))
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        alpha = self.alpha

        # Su_old is a numerically-computed posterior covariance and can be (near-)singular,
        # which makes every tf.linalg.solve(Saa,.) fail ("Input matrix is not
        # invertible"). Regularise it like the other covariance blocks.
        Saa = gpflow.utilities.add_noise_cov(self.Su_old, jitter)
        ma = self.mu_old
        # a is old inducing points, b is new
        # f is training points
        # s is test points
        Kfdiag = self.kernel(self.X, full_cov=False)
        Kbf = covariances.Kuf(self.inducing_variable, self.kernel, self.X)
        Kbb = covariances.Kuu(self.inducing_variable, self.kernel, jitter=jitter)
        Kba = covariances.Kuf(self.inducing_variable, self.kernel, self.Z_old)
        Kab = tf.linalg.matrix_transpose(Kba)
        Kaa_cur = gpflow.utilities.add_noise_cov(self.kernel(self.Z_old), jitter)
        Kaa = gpflow.utilities.add_noise_cov(self.Kaa_old, jitter)

        err = self.Y - self.mean_function(self.X)
        Lb = tf.linalg.cholesky(Kbb)
        Lbinv_Kbf = tf.linalg.triangular_solve(Lb, Kbf, lower=True)

        Qff_diag = tf.reduce_sum(tf.square(Lbinv_Kbf), axis=0)
        Dff = sigma2 + alpha * (Kfdiag - Qff_diag)
        Lbinv_Kbf_LDff = Lbinv_Kbf / tf.sqrt(Dff)
        d1 = tf.matmul(Lbinv_Kbf_LDff, Lbinv_Kbf_LDff, transpose_b=True)

        Lbinv_Kba = tf.linalg.triangular_solve(Lb, Kba, lower=True)
        Kab_Lbinv = tf.linalg.matrix_transpose(Lbinv_Kba)
        Sainv_Kab_Lbinv = tf.linalg.solve(Saa, Kab_Lbinv)
        Kainv_Kab_Lbinv = tf.linalg.solve(Kaa, Kab_Lbinv)
        Da_Kab_Lbinv = Sainv_Kab_Lbinv - Kainv_Kab_Lbinv
        d2 = tf.matmul(Lbinv_Kba, Da_Kab_Lbinv)

        Kaadiff = Kaa_cur - tf.matmul(Kab_Lbinv, Lbinv_Kba)
        LM = tf.linalg.cholesky(Kaadiff)
        LMT = tf.linalg.matrix_transpose(LM)
        Sainv_LM = tf.linalg.solve(Saa, LM)
        Kainv_LM = tf.linalg.solve(Kaa, LM)
        SK_LM = Sainv_LM - Kainv_LM
        LMT_SK_LM = tf.matmul(LMT, SK_LM)
        Q = tf.eye(Ma, dtype=gpflow.default_float()) + alpha * LMT_SK_LM
        LQ = tf.linalg.cholesky(Q)

        LMT_Da_Kab_Lbinv = tf.matmul(LMT, Da_Kab_Lbinv)
        Qinv_t1 = tf.linalg.solve(Q, LMT_Da_Kab_Lbinv)
        t1_Qinv_t1 = tf.matmul(LMT_Da_Kab_Lbinv, Qinv_t1, transpose_a=True)
        d3 = - alpha * t1_Qinv_t1

        D = tf.eye(Mb, dtype=gpflow.default_float()) + d1 + d2 + d3
        D = gpflow.utilities.add_noise_cov(D, jitter)
        LD = tf.linalg.cholesky(D)

        Sainv_ma = tf.linalg.solve(Saa, ma)
        LMT_Sainv_ma = tf.matmul(LMT, Sainv_ma)
        Lbinv_Kba_Da = tf.linalg.matrix_transpose(Da_Kab_Lbinv)
        Lbinv_Kba_Da_LM = tf.matmul(Lbinv_Kba_Da, LM)
        Qinv_LMT_Sainv_ma = tf.linalg.solve(Q, LMT_Sainv_ma)
        Sinv_y = self.Y / tf.reshape(Dff, [self.num_data, 1])
        c1 = tf.matmul(Lbinv_Kbf, Sinv_y)
        c2 = tf.matmul(Lbinv_Kba, Sainv_ma)
        c3 = - alpha * tf.matmul(Lbinv_Kba_Da_LM, Qinv_LMT_Sainv_ma)
        c = c1 + c2 + c3

        LDinv_c = tf.linalg.triangular_solve(LD, c, lower=True)
        LSa = tf.linalg.cholesky(Saa)
        La = tf.linalg.cholesky(Kaa)

        return (Kbf, Kba, Kaa, Kaa_cur, LSa, La, Kbb, Lb, D, LD,
                Lbinv_Kba, LDinv_c, err, Dff, Kaadiff, Sainv_ma, Q, LQ, LM)

    def maximum_log_likelihood_objective(self):
        Mb = self.inducing_variable.num_inducing
        Ma = self.M_old
        jitter = gpflow.utilities.to_default_float(max(gpflow.config.default_jitter(), 1e-4))
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        alpha = self.alpha
        # Su_old is a numerically-computed posterior covariance and can be (near-)singular,
        # which makes every tf.linalg.solve(Saa,.) fail ("Input matrix is not
        # invertible"). Regularise it like the other covariance blocks.
        Saa = gpflow.utilities.add_noise_cov(self.Su_old, jitter)
        ma = self.mu_old
        N = self.num_data

        # a is old inducing points, b is new
        # f is training points
        (Kbf, Kba, Kaa, Kaa_cur, LSa, La, Kbb, Lb, D, LD, Lbinv_Kba, LDinv_c,
            err, Dff, Kaadiff, Sainv_ma, Q, LQ, LM) = self._common_terms()

        Lainv_ma = tf.linalg.triangular_solve(LSa, ma, lower=True)

        # constant term
        bound = -0.5 * N * np.log(2 * np.pi)
        # quadratic term
        bound += -0.5 * tf.reduce_sum(tf.square(err) / tf.reshape(Dff, [N, 1]))
        bound += -0.5 * tf.reduce_sum(tf.square(Lainv_ma))
        bound += 0.5 * tf.reduce_sum(tf.square(LDinv_c))
        ma_Sainv_LM_transposed = tf.matmul(LM, Sainv_ma, transpose_a=True)  # transpose(transpose(Sainv_ma) @ LM) = transpose(LM) @ Sainv_ma
        Qinv_LM_Sainv_ma = tf.linalg.solve(Q, ma_Sainv_LM_transposed)
        bound += 0.5 * alpha * \
            tf.reduce_sum(tf.matmul(ma_Sainv_LM_transposed, Qinv_LM_Sainv_ma, transpose_a=True))

        # log det term
        bound += -0.5 * tf.reduce_sum(tf.math.log(Dff))
        bound += - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LD)))

        # delta 1: trace-like term
        bound += - 0.5 * (1 - alpha) / alpha * \
            tf.reduce_sum(tf.math.log(Dff / sigma2))

        # delta 2
        bound += - 1.0 / alpha * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LQ)))
        bound += tf.reduce_sum(tf.math.log(tf.linalg.diag_part(La)))
        bound += - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSa)))

        return bound

    def predict_f(self, Xnew, full_cov=False):
        # configurable through gpflow's default jitter so a failed fit can be retried with
        # a larger value (Extrapolation._optimize_with_jitter). Never below the 1e-4 the
        # power ep models were tuned with
        jitter = gpflow.utilities.to_default_float(max(gpflow.config.default_jitter(), 1e-4))

        # a is old inducing points, b is new
        # f is training points
        # s is test points
        Kbs = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)
        (Kbf, Kba, Kaa, Kaa_cur, LSa, La, Kbb, Lb, D, LD, Lbinv_Kba, LDinv_c,
            err, Dff, Kaadiff, Sainv_ma, Q, LQ, LM) = self._common_terms()

        Lbinv_Kbs = tf.linalg.triangular_solve(Lb, Kbs, lower=True)
        LDinv_Lbinv_Kbs = tf.linalg.triangular_solve(LD, Lbinv_Kbs, lower=True)
        mean = tf.matmul(LDinv_Lbinv_Kbs, LDinv_c, transpose_a=True)

        if full_cov:
            Kss = self.kernel(Xnew) + jitter * tf.eye(tf.shape(Xnew)[0], dtype=gpflow.default_float())
            var1 = Kss
            var2 = - tf.matmul(Lbinv_Kbs, Lbinv_Kbs, transpose_a=True)
            var3 = tf.matmul(LDinv_Lbinv_Kbs, LDinv_Lbinv_Kbs, transpose_a=True)
            var = var1 + var2 + var3
        else:
            var1 = self.kernel(Xnew, full_cov=False)
            var2 = -tf.reduce_sum(tf.square(Lbinv_Kbs), axis=0)
            var3 = tf.reduce_sum(tf.square(LDinv_Lbinv_Kbs), axis=0)
            var = var1 + var2 + var3

        return mean + self.mean_function(Xnew), var
