# Copyright 2019-2022 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-spots/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Variational Inference functions for spot decoding. Code adapted from PoSTcode
https://github.com/gerstung-lab/postcode (https://doi.org/10.1101/2021.10.12.464086)."""

import scipy
import torch
import numpy as np
from tqdm import tqdm
import pyro
from pyro.distributions import (RelaxedBernoulli, Categorical, constraints,
                                MultivariateNormal)
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from torch.autograd import Function

assert pyro.__version__.startswith('1')


def reshape_torch_array(torch_array):
    """Reshape a ``[k, r, c]`` array into a ``[k, r * c]``.

    Args:
        torch_array (torch.tensor): Array to be reshaped.

    Returns:
        torch.tensor: Reshaped array.
    """
    return torch_array.transpose(1, 2).reshape(torch_array.shape[0], -1)


def normalize_spot_values(data):
    """Normalizes spot intensity data array such that log of data has a mean of zero and standard
    deviation of one.
    
    Args:
        data (torch.tensor): Input data formatted as torch array with shape ``[num_spots, r * c]``.
    
    Returns:
        torch.tensor: Normalized data array.
    """
    # TODO: add clipping functionality

    s = torch.tensor(np.percentile(data.cpu().numpy(), 60, axis=0))
    max_s = torch.tensor(np.percentile(data.cpu().numpy(), 99.9, axis=0))
    min_s = torch.min(data, dim=0).values
    eps = 1e-6*max_s
    log_add = (s ** 2 - max_s * min_s) / (max_s + min_s - 2 * s)
    log_add = torch.max(-torch.min(data, dim=0).values + eps,
                        other=log_add.float() + eps)
    data_log = torch.log10(data + log_add)
    data_log_mean = data_log.mean(dim=0, keepdim=True)
    data_log_std = data_log.std(dim=0, keepdim=True)
    data_norm = (data_log - data_log_mean) / data_log_std  # column-wise normalization

    return data_norm


def kronecker_product(a, b):
    """Matrix multiplication with two matrices of arbitrary size.

    Args:
        a (torch.tensor): Matrix of arbitrary size.
        b (torch.tensor): Matrix of arbitrary size.
    
    Returns:
        torch.tensor: Kronecker product of ``a`` and ``b``.
    """
    a_height, a_width = a.size()
    b_height, b_width = b.size()
    out_height = a_height * b_height
    out_width = a_width * b_width
    tiled_b = b.repeat(a_height, a_width)
    expanded_a = (a.unsqueeze(2).unsqueeze(3).repeat(1, b_height, b_width, 1).view(
        out_height, out_width))

    return expanded_a * tiled_b


def chol_sigma_from_vec(sigma_vec, dim):
    L = torch.zeros(dim, dim)
    L[torch.tril(torch.ones(dim, dim)) == 1] = sigma_vec

    return torch.mm(L, torch.t(L))


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input

mat_sqrt_ = MatrixSquareRoot.apply
def mat_sqrt(A):
    return mat_sqrt_(A)


def instantiate_rb_params(r, c, codes, params_mode):
    """Instantiates parameters for model of mixture of Relaxed Bernoulli distributions.

    Args:
        r (int): Number of rounds.
        c (int): Number of channels.
        codes (torch.tensor): Codebook formatted as torch array with shape
            ``[num_barcodes + 1, r * c]``.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds. valid options: ['2', '2*R', '2*C', '2*R*C'].

    Returns:
        scaled_sigma (torch.tensor): Sigma parameter of Relaxed Bernoulli.
        aug_temperature (torch.tensor): Temperature parameter of Relaxed Bernoulli.
    """

    if params_mode == '2':
        # two param - one for 0-channel, one for 1-channel
        sigma = pyro.param("sigma", torch.ones(torch.Size(
            [2])) * 0.5, constraint=constraints.unit_interval)
        aug_sigma = torch.gather(
            sigma, 0, codes.reshape(-1).long()).reshape(codes.shape)
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma  # is 0.3 an arbitrary choice?
        temperature = pyro.param('temperature', torch.ones(
            torch.Size([2])) * 0.5, constraint=constraints.unit_interval)
        aug_temperature = torch.gather(
            temperature, 0, codes.reshape(-1).long()).reshape(codes.shape)
    elif params_mode == '2*R':
        # 2*R params
        sigma = pyro.param("sigma", torch.ones(torch.Size(
            [2, r])) * 0.5, constraint=constraints.unit_interval)
        sigma_temp = sigma.unsqueeze(-1).repeat(1, 1, c)
        sigma_temp1 = reshape_torch_array(sigma_temp)
        aug_sigma = torch.gather(sigma_temp1, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature = pyro.param("temperature", torch.ones(
            torch.Size([2, r])) * 0.5, constraint=constraints.unit_interval)
        temperature_temp = temperature.unsqueeze(-1).repeat(1, 1, c)
        temperature_temp1 = reshape_torch_array(temperature_temp)
        aug_temperature = torch.gather(temperature_temp1, 0, codes.long())
    elif params_mode == '2*C':
        # 2*C params
        sigma = pyro.param("sigma", torch.ones(torch.Size(
            [2, c])) * 0.5, constraint=constraints.unit_interval)
        sigma_temp = sigma.unsqueeze(-1).repeat(1, 1, r)
        sigma_temp1 = reshape_torch_array(sigma_temp)
        aug_sigma = torch.gather(sigma_temp1, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature = pyro.param("temperature", torch.ones(
            torch.Size([2, c])) * 0.5, constraint=constraints.unit_interval)
        temperature_temp = temperature.unsqueeze(-1).repeat(1, 1, r)
        temperature_temp1 = reshape_torch_array(temperature_temp)
        aug_temperature = torch.gather(temperature_temp1, 0, codes.long())
    elif params_mode == '2*R*C':
        # 2*R*C params
        sigma = pyro.param("sigma", torch.ones(torch.Size(
            [2, r * c])) * 0.5, constraint=constraints.unit_interval)
        aug_sigma = torch.gather(sigma, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature = pyro.param("temperature", torch.ones(
            torch.Size([2, r * c])) * 0.5, constraint=constraints.unit_interval)
        aug_temperature = torch.gather(temperature, 0, codes.long())
    else:
        assert False, "%s not supported" % params_mode

    return scaled_sigma, aug_temperature


def instantiate_gaussian_params(r, c, codes):
    """Instantiates parameters for model of mixture of Gaussian distributions.

    Args:
        r (int): Number of rounds.
        c (int): Number of channels.
        codes (torch.tensor): Codebook formatted as torch array with shape
            ``[num_barcodes + 1, r * c]``.

    Returns:
        theta (torch.tensor): Mean parameter of Gaussian.
        sigma (torch.tensor): Covariance parameter of Gaussian.
    """
    d = c * r

    # using tensor sigma
    sigma_c_v = pyro.param('sigma_c_v', torch.eye(c)[np.tril_indices(c, 0)])
    sigma_c = chol_sigma_from_vec(sigma_c_v, c)
    sigma_r_v = pyro.param('sigma_r_v', torch.eye(d)[np.tril_indices(r, 0)])
    sigma_r = chol_sigma_from_vec(sigma_r_v, r)
    sigma = kronecker_product(sigma_r, sigma_c)

    codes_tr_v = pyro.param('codes_tr_v', 3 * torch.ones(1, d), constraint=constraints.greater_than(1.))
    codes_tr_consts_v = pyro.param('codes_tr_consts_v', -1 * torch.ones(1, d))

    theta = torch.matmul(codes * codes_tr_v + codes_tr_consts_v, mat_sqrt(sigma))

    return theta, sigma


@config_enumerate
def model_constrained_tensor(
        data,
        codes,
        c,
        r,
        batch_size=None,
        params_mode='2*R*C'):
    """Model definition: relaxed bernoulli, paramters are shared across all genes, but might
    differ across channels or rounds.

    Args:
        data (torch.tensor): Input data formatted as torch array with shape ``[num_spots, r * c]``.
        codes (torch.tensor): Codebook formatted as torch array with shape
            ``[num_barcodes + 1, r * c]``.
        c (int): Number of channels.
        r (int): Number of rounds.
        batch_size (int): Size of batch for training.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds for model of Relaxed Bernoulli distributions, or model of Gaussians.
            Valid options: ['2', '2*R', '2*C', '2*R*C', 'Gaussian']. Defaults to '2*R*C'. 

    Returns:
        None
    """
    k = codes.shape[0]
    w = pyro.param('weights', torch.ones(k) / k, constraint=constraints.simplex)

    if params_mode in ['2', '2*R', '2*C', '2*R*C']:
        scaled_sigma, aug_temperature = instantiate_rb_params(r, c, codes, params_mode)

        with pyro.plate('data', data.shape[0], batch_size) as batch:
            z = pyro.sample('z', Categorical(w))
            pyro.sample(
                'obs',
                RelaxedBernoulli(
                    temperature=aug_temperature[z],
                    probs=scaled_sigma[z]).to_event(1),
                obs=data[batch])

    elif params_mode == 'Gaussian':
        theta, sigma = instantiate_gaussian_params(r, c, codes)

        with pyro.plate('data', data.shape[0], batch_size) as batch:
            z = pyro.sample('z', Categorical(w))
            pyro.sample(
                'obs',
                MultivariateNormal(
                    loc=theta[z],
                    covariance_matrix=sigma),
                obs=data[batch])

    else:
        assert False, "%s not supported" % params_mode


def train(svi, num_iter, data, codes, c, r, batch_size, params_mode):
    """Do the training for SVI model.

    Args:
        svi (pyro.infer.SVI): stochastic variational inference model.
        num_iter (int): Number of iterations for training.
        data (torch.tensor): Input data formatted as torch array with shape ``[num_spots, r * c]``.
        codes (torch.tensor): Codebook formatted as torch array with shape
            ``[num_barcodes + 1, r * c]``.
        c (int): Number of channels.
        r (int): Number of rounds.
        batch_size (int): Size of batch for training.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds for model of Relaxed Bernoulli distributions, or model of Gaussians.
            Valid options: ['2', '2*R', '2*C', '2*R*C', 'Gaussian']. Defaults to '2*R*C'. 

    Returns:
        list: losses.

    """
    pyro.clear_param_store()
    losses = []
    for _ in tqdm(range(num_iter)):
        loss = svi.step(data, codes, c, r, batch_size, params_mode)
        losses.append(loss)
    return losses


def rb_e_step(data, codes, w, temperature, sigma, c, r, params_mode='2*R*C'):
    """Estimate the posterior probability for spot assignment from a mixture of Relaxed
    Bernoulli distributions.

    Args:
        data (torch.tensor): Input data formatted as torch array with shape ``[num_spots, r * c]``.
        codes (torch.tensor): Codebook formatted as torch array with shape
            ``[num_barcodes + 1, r * c]``.
        w (torch.array): Weight parameter with length ``num_barcodes + 1``.
        temperature (torch.array): Temperature parameter for Relaxed Bernoulli, shape depends on
             `params_mode`.
        sigma (torch.array): Sigma parameter for Relaxed Bernoulli, shape depends on `params_mode`.
        c (int): Number of channels.
        r (int): Number of rounds.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds. Valid options: ['2', '2*R', '2*C', '2*R*C'].

    Returns:
        normalized class probability with shape ``[num_spots, num_barcodes + 1]``.
    """
    k = codes.shape[0]  # num_barcodes + 1
    class_logprobs = np.ones((data.shape[0], k))

    if params_mode == '2':  # two params
        aug_sigma = torch.gather(
            sigma, 0, codes.reshape(-1).long()).reshape(codes.shape)
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        aug_temperature = torch.gather(
            temperature, 0, codes.reshape(-1).long()).reshape(codes.shape)
    elif params_mode == '2*R':  # 2*R params
        sigma_temp = sigma.unsqueeze(-1).repeat(1, 1, c)
        sigma_temp1 = reshape_torch_array(sigma_temp)
        aug_sigma = torch.gather(sigma_temp1, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature_temp = temperature.unsqueeze(-1).repeat(1, 1, c)
        temperature_temp1 = reshape_torch_array(temperature_temp)
        aug_temperature = torch.gather(temperature_temp1, 0, codes.long())
    elif params_mode == '2*C':  # 2*C params
        sigma_temp = sigma.unsqueeze(-1).repeat(1, 1, r)
        sigma_temp1 = reshape_torch_array(sigma_temp)
        aug_sigma = torch.gather(sigma_temp1, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature_temp = temperature.unsqueeze(-1).repeat(1, 1, r)
        temperature_temp1 = reshape_torch_array(temperature_temp)
        aug_temperature = torch.gather(temperature_temp1, 0, codes.long())
    elif params_mode == '2*R*C':  # 2*R*C params
        aug_sigma = torch.gather(sigma, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        aug_temperature = torch.gather(temperature, 0, codes.long())
    else:
        assert False, "%s not supported" % params_mode

    batch_sz = 50000
    for idx in range(len(data) // batch_sz + 1):
        ind_start = idx * batch_sz
        ind_end = torch.min(torch.tensor([(idx + 1) * batch_sz, len(data)]))
        for idx in tqdm(range(k)):
            dist = RelaxedBernoulli(
                temperature=aug_temperature[idx],
                probs=scaled_sigma[idx]).to_event(1)
            class_logprobs[ind_start:ind_end, idx] = (
                w[idx].log() + dist.log_prob(data[ind_start:ind_end])).cpu().numpy()

    # basically doing a stable_softmax here
    numerator = np.exp(class_logprobs - np.max(class_logprobs, axis=1)[:, None])
    class_prob_norm = np.divide(numerator, np.sum(numerator, axis=1)[:, None])

    return class_prob_norm


def gaussian_e_step(data, w, theta, sigma, K):
    """Estimate the posterior probability for spot assignment from a mixture of Gaussian
    distributions.

    Args:
        data (torch.tensor): Input data formatted as torch array with shape ``[num_spots, r * c]``.
        w (torch.tensor): Weight parameter with length ``num_barcodes + 1``.
        theta (torch.tensor): Mean parameter for Gaussian distribution.
        sigma (torch.tensor): Covariance parameter for Gaussian distribution.
        K (torch.tensor): Number of rounds * number of channels.

    Returns:
        normalized class probability with shape ``[num_spots, num_barcodes + 1]``.

    """
    class_probs = torch.ones(data.shape[0], K)
    batch_sz = 50000

    for idx in range(len(data) // batch_sz + 1):
        ind_start  = idx * batch_sz
        ind_end = torch.min(torch.tensor([(idx+1) * batch_sz, len(data)]))
        for k in tqdm(range(K)):
            dist = MultivariateNormal(theta[k], sigma)
            class_probs[ind_start:ind_end, k] = w[k] * torch.exp(dist.log_prob(data))

    class_prob_norm = class_probs.div(torch.sum(class_probs, dim=1, keepdim=True)).cpu().numpy()

    return class_prob_norm


def decoding_function(spots,
                      barcodes,
                      num_iter=500,
                      batch_size=15000,
                      set_seed=1,
                      params_mode='2*R*C'):
    """Main function for the spot decoding.

    Args:
        spots (numpy.array): Input spot intensity array with shape ``[num_spots, c, r]``.
        barcodes (numpy.array): Input codebook array with shape ``[num_barcodes, c, r]``.
        num_iter (int): Number of iterations for training. Defaults to 500.
        batch_size (int): Size of batch for training. Defaults to 15000.
        set_seed (int): Seed for randomness. Defaults to 1.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds for model of Relaxed Bernoulli distributions, or model of Gaussians.
            Valid options: ['2', '2*R', '2*C', '2*R*C', 'Gaussian']. Defaults to '2*R*C'. 

    Returns:
        results (dict): The decoding results as a dictionary: 'class_probs': posterior
            probabilities for each spot and each gene category; 'params': estimated model
            parameters.
    """
    # if cuda available, runs on gpu
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    valid_params_modes = ['2', '2*R', '2*C', '2*R*C', 'Gaussian']
    if params_mode not in valid_params_modes:
            raise ValueError('Invalid params_mode supplied: {}. '
                             'Must be one of {}'.format(params_mode,
                                                        valid_params_modes))

    num_spots, c, r = spots.shape

    data = reshape_torch_array(torch.tensor(spots).float())
    codes = reshape_torch_array(torch.tensor(barcodes).float())

    if params_mode=='Gaussian':
        data = normalize_spot_values(data)
        auto_guide_constrained_tensor = AutoDelta(poutine.block(model_constrained_tensor,
                                                  expose=['weights',
                                                          'codes_tr_v',
                                                          'codes_tr_consts_v',
                                                          'sigma_c_v',
                                                          'sigma_r_v']))

    else:
        auto_guide_constrained_tensor = AutoDelta(poutine.block(model_constrained_tensor,
                                                  expose=['weights',
                                                          'temperature',
                                                          'sigma']))

    optim = Adam({'lr': 0.085, 'betas': [0.85, 0.99]})
    svi = SVI(model_constrained_tensor, auto_guide_constrained_tensor,
              optim, loss=TraceEnum_ELBO(max_plate_nesting=1))
    pyro.set_rng_seed(set_seed)

    losses = train(svi, num_iter, data, codes, c, r,
                   min(num_spots, batch_size), params_mode)

    if params_mode=='Gaussian':
        w_star = pyro.param('weights').detach()

        sigma_c_v_star = pyro.param('sigma_c_v').detach()
        sigma_r_v_star = pyro.param('sigma_r_v').detach()
        sigma_r_star = chol_sigma_from_vec(sigma_r_v_star, r)
        sigma_c_star = chol_sigma_from_vec(sigma_c_v_star, c)
        sigma_star = kronecker_product(sigma_r_star, sigma_c_star)

        codes_tr_v_star = pyro.param('codes_tr_v').detach()
        codes_tr_consts_v_star = pyro.param('codes_tr_consts_v').detach()
        theta_star = torch.matmul(codes * codes_tr_v_star + codes_tr_consts_v_star, mat_sqrt(sigma_star))

        class_probs_star = gaussian_e_step(data, w_star, theta_star, sigma_star, K=codes.shape[0])

        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.FloatTensor")

        torch_params = {
            'w_star': w_star.cpu(),
            'sigma_star': sigma_star.cpu(),
            'sigma_r_star': sigma_r_star.cpu(),
            'sigma_c_star': sigma_c_star.cpu(),
            'theta_star': theta_star.cpu(),
            'codes_tr_consts_v_star': codes_tr_consts_v_star.cpu(),
            'codes_tr_v_star': codes_tr_v_star.cpu(),
            'losses': losses
        }
    
    else:

        w_star = pyro.param('weights').detach()
        temperature_star = pyro.param('temperature').detach()
        sigma_star = pyro.param('sigma').detach()

        class_probs_star = rb_e_step(
            data, codes, w_star, temperature_star, sigma_star, c, r, params_mode)

        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.FloatTensor")

        torch_params = {
            'w_star': w_star.cpu(),
            'temperature_star': temperature_star.cpu(),
            'sigma_star': sigma_star.cpu(),
            'losses': losses
        }

    results = {'class_probs': class_probs_star,
               'params': torch_params}

    return results
