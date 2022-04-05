import GPy
from matplotlib import pyplot as plt
import numpy as np
from sklearn.covariance import log_likelihood
from warpingFunction import WarpingFunction
from multiprocessing.pool import ThreadPool
from GPy.core.parameterization.variational import NormalPosterior
from inference import NewInference
from GPy import likelihoods
log_2_pi = np.log(2*np.pi)

class WarpedModelSimple(GPy.models.SparseGPRegression):
    def __init__(self, X, Y, kernel=None, Z=None, num_inducing=10, X_variance=None, mean_function=None, normalizer=None, mpi_comm=None, name='SparseModel', warping_function=None, likelihood=None, inference_method=None, Y_metadata=None):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        num_data, input_dim = X.shape
        self.Y_untransformed = Y.copy()
        self.Y_Transformed = None
        self.Y_part = (0.5 * Y.astype(np.float64) + 0.5)
        
        self.kernel = kernel

        self.warping_function = WarpingFunction()

        # Z defaults to a subset of the data
        if Z is None:
            i = np.random.permutation(num_data)[:min(num_inducing, num_data)]
            Z = X.view(np.ndarray)[i].copy()
        else:
            assert Z.shape[1] == input_dim

        likelihood = likelihoods.Gaussian()

        if not (X_variance is None):
            X = NormalPosterior(X,X_variance)

        infr = NewInference()

        super(GPy.models.SparseGPRegression, self).__init__(X, Y=self.transform_data(), Z=Z, kernel=kernel, likelihood=likelihood, mean_function=mean_function,
        inference_method=infr, normalizer=normalizer, mpi_comm=mpi_comm, name=name)

        self.Y_normalized = self.Y_normalized.copy()

    def transform_data(self):
        """
        Pass the input Y values through the warping function and store them in Y_Transformed
        """
        if self.Y_Transformed is None:
            y1 = self.warping_function.f(self.Y_part[:, 0].copy()).copy()
            y2 = self.warping_function.f(self.Y_part[:, 1].copy()).copy()
            y3 = self.warping_function.f(self.Y_part[:, 2].copy()).copy()
            y4 = self.warping_function.f(self.Y_part[:, 3].copy()).copy()
            y5 = self.warping_function.f(self.Y_part[:, 4].copy()).copy()
            y6 = self.warping_function.f(self.Y_part[:, 5].copy()).copy()
            self.Y_Transformed = np.column_stack((y1, y2, y3, y4, y5, y6))
        return self.Y_Transformed

    def _get_warped_term(self, mean, std, gh_samples, pred_init=None):
        arg1 = gh_samples.dot(std.T) * np.sqrt(2)
        arg2 = np.ones(shape=gh_samples.shape).dot(mean.T)
        val = self.warping_function.f_inv(arg1 + arg2, y=pred_init)
        return val

    def _get_warped_mean(self, mean, std, pred_init=None, deg_gauss_hermite=20):
        """
        Calculate the warped mean by using Gauss-Hermite quadrature.
        """
        gh_samples, gh_weights = np.polynomial.hermite.hermgauss(deg_gauss_hermite)
        gh_samples = gh_samples[:, None]
        gh_weights = gh_weights[None, :]
        return gh_weights.dot(self._get_warped_term(mean, std, gh_samples)) / np.sqrt(np.pi)
    
    def _get_warped_variance(self, mean, std, pred_init=None, deg_gauss_hermite=20):
        """
        Calculate the warped variance by using Gauss-Hermite quadrature.
        """
        gh_samples, gh_weights = np.polynomial.hermite.hermgauss(deg_gauss_hermite)
        gh_samples = gh_samples[:, None]
        gh_weights = gh_weights[None, :]
        arg1 = gh_weights.dot(self._get_warped_term(mean, std, gh_samples, 
                                                    pred_init=pred_init) ** 2) / np.sqrt(np.pi)
        arg2 = self._get_warped_mean(mean, std, pred_init=pred_init,
                                     deg_gauss_hermite=deg_gauss_hermite)
        return arg1 - (arg2 ** 2)

    def predict(self, Xnew, likelihood=None, Y_metadata=None):
        """Prediction on the new data

        Parameters
        ----------
        Xnew : array_like, shape = (n_samples, n_features)
            The test data.

        Returns
        -------
        mean : array_like, shape = (n_samples, output.dim)
            Posterior mean at the location of Xnew

        var : array_like, shape = (n_samples, 1)
            Posterior variance at the location of Xnew
        """
 
        mean, var = super(WarpedModelSimple, self).predict(Xnew, kern=self.kernel, full_cov=False)
        std = np.sqrt(var)
        Wmean = self._get_warped_mean(mean, std, pred_init=None,
                            deg_gauss_hermite=20).T
        Wvar = self._get_warped_variance(mean, std, pred_init=None,
                                            deg_gauss_hermite=20).T
        Wmean = 2 * Wmean - 1
        WvarNew = 2 * Wvar - 1
        return Wmean, Wvar, WvarNew

    def calculate_log(self):
        """
        legacy function
        """
        m = self.mean_function.f(self.X)
        variance = self.likelihood.gaussian_variance(self.Y_metadata)
        for annotator in annotators:
            YYT_factor = annotator-m
            Ky = self.kern.K(self.X).copy()
            GPy.util.diag.add(Ky, variance+1e-8)

            Wi, LW, LWi, W_logdet = GPy.util.linalg.pdinv(Ky)

            alpha, _ = GPy.util.linalg.dpotrs(LW, YYT_factor, lower=1)

            log_marginal =  0.5*(-annotator.size * log_2_pi - annotator.shape[1] * W_logdet - np.sum(alpha * YYT_factor))
            jacobian = self.warping_function.fgrad_y(self.transform_data())
            total = total + log_marginal + np.log(jacobian).sum()
        self._log_marginal_likelihood = total

    def log_likelihood(self):
        """
        Compute the Log marginal likelihood taking into account the jacobian terms of warping
        """        
        logLikelihood = GPy.core.GP.log_likelihood(self)
        Yval = self.transform_data()
        # return logLikelihood + np.log(self.warping_function.fgrad_y(Yval)).sum()
        jacobian1 = self.warping_function.fgrad_y(Yval[:, 0])
        jacobian2 = self.warping_function.fgrad_y(Yval[:, 1])
        jacobian3 = self.warping_function.fgrad_y(Yval[:, 2])
        jacobian4 = self.warping_function.fgrad_y(Yval[:, 3])
        jacobian5 = self.warping_function.fgrad_y(Yval[:, 4])
        jacobian6 = self.warping_function.fgrad_y(Yval[:, 5])
        return logLikelihood + np.log(jacobian1).sum() + np.log(jacobian2).sum() + np.log(jacobian3).sum() + np.log(jacobian4).sum() + np.log(jacobian5).sum() + np.log(jacobian6).sum()

    def _build_from_input_dict(input_dict, data=None):
        input_dict = GPy.core.SparseGP._format_input_dict(input_dict, data)
        input_dict.pop('class')
        #input_dict['warp'] = False
        #print(input_dict)       
        return WarpedModelSimple(**input_dict)

    def load_model(output_filename, data=None):
        compress = output_filename.split(".")[-1] == "zip"
        import json
        if compress:
            import gzip
            with gzip.GzipFile(output_filename, 'r') as json_data:
                json_bytes = json_data.read()
                json_str = json_bytes.decode('utf-8')
                output_dict = json.loads(json_str)
        else:
            with open(output_filename) as json_data:
                output_dict = json.load(json_data)
        import copy
        output_dict = copy.deepcopy(output_dict)
        output_dict["name"] = str(output_dict["name"])
        return WarpedModelSimple._build_from_input_dict(output_dict, data)

    def predict_quantiles(self, X, quantiles=(2.5, 97.5), Y_metadata=None, likelihood=None, kern=None):
        """
        Get the predictive quantiles around the prediction at X

        :param X: The points at which to make a prediction
        :type X: np.ndarray (Xnew x self.input_dim)
        :param quantiles: tuple of quantiles, default is (2.5, 97.5) which is the 95% interval
        :type quantiles: tuple
        :returns: list of quantiles for each X and predictive quantiles for interval combination
        :rtype: [np.ndarray (Xnew x self.input_dim), np.ndarray (Xnew x self.input_dim)]
        """
        
        qs = super(WarpedModelSimple, self).predict_quantiles(X, quantiles, Y_metadata=Y_metadata, likelihood=likelihood, kern=kern)
        X = [self.warping_function.f_inv(q) for q in qs]
        X[0] = X[0].flatten()
        X[1] = X[1].flatten()
        X[0] = [x * 2 - 1 for x in X[0]]
        X[1] = [x * 2 - 1 for x in X[1]]
        return X

    def to_dict(self, save_data=True):
        input_dict = super(WarpedModelSimple, self).to_dict(save_data)
        try:
            input_dict["Y"] = self.Y_untransformed.values.tolist()
        except:
            input_dict["Y"] = self.Y_untransformed.tolist()
        return input_dict