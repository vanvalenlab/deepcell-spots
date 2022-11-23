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

"""Variational Inference functions for spot decoding"""

import numpy as np
import torch
from tqdm import tqdm
import pyro
from pyro.distributions import RelaxedBernoulli, Categorical, constraints
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro import poutine
from pyro.infer.autoguide import AutoDelta

assert pyro.__version__.startswith('1')


def reshape_torch_array(torch_array):
    """Reshape a ``[k, r, c]`` array into a ``[k, r * c]``.

    Args:
        torch_array (torch.array): Array to be reshaped.

    Returns:
        torch.array: Reshaped array.
    """
    D = torch_array.shape[1] * torch_array.shape[2]
    return torch_array.transpose(1, 2).reshape(torch_array.shape[0], D)


@config_enumerate
def model_constrained_tensor(data, codes, c, r, batch_size=None, params_mode='2*R*C'):
    """Model definition: relaxed bernoulli, paramters are shared across all genes, but might differ across channels or rounds.

    Args:
        data (torch.array): Input data formatted as torch array with shape ``[num_spots, r * c]``.
        codes (torch.array): Codebook formatted as torch array with shape ``[num_barcodes + 1, r * c]``.
        c (int): Number of channels.
        r (int): Number of rounds.
        batch_size (int): Size of batch for training. Defaults to 1000.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds. valid options: {'1', '2', '2*R', '2*C', '2*R*C'}.

    Returns:
        None
    """
    k = codes.shape[0]
    w = pyro.param('weights', torch.ones(k) / k,
                   constraint=constraints.simplex)

    if params_mode == '1':
        # one param
        sigma = pyro.param("sigma", torch.ones(torch.Size(
            [1])) * 0.5, constraint=constraints.unit_interval)
        scaled_sigma = codes + (-1)**codes * 0.3 * sigma
        temperature = pyro.param("temperature", torch.ones(
            torch.Size([1])) * 0.5, constraint=constraints.unit_interval)
        aug_temperature = temperature.unsqueeze(-1).repeat(k, 1)
    elif params_mode == '2':
        # two param - one for 0-channel, one for 1-channel
        sigma = pyro.param("sigma", torch.ones(torch.Size(
            [2])) * 0.5, constraint=constraints.unit_interval)
        aug_sigma = torch.gather(
            sigma, 0, codes.reshape(-1).long()).reshape(codes.shape)
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
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

    with pyro.plate('data', data.shape[0], batch_size) as batch:
        z = pyro.sample('z', Categorical(w))
        pyro.sample('obs', RelaxedBernoulli(
            temperature=aug_temperature[z], probs=scaled_sigma[z]).to_event(1), obs=data[batch])


# Initialize an auto guide on the model
auto_guide_constrained_tensor = AutoDelta(poutine.block(
    model_constrained_tensor, expose=['weights', 'temperature', 'sigma']))


def train(svi, num_iter, data, codes, c, r, batch_size, params_mode):
    """Do the training for SVI model.
    Args:
        svi (pyro.infer.SVI): stochastic variational inference model.
        num_iter (int): Number of iterations for training. Defaults to 200.
        data (torch.array): Input data formatted as torch array with shape ``[num_spots, r * c]``.
        codes (torch.array): Codebook formatted as torch array with shape ``[num_barcodes + 1, r * c]``.
        c (int): Number of channels.
        r (int): Number of rounds.
        batch_size (int): Size of batch for training. Defaults to 1000.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds. valid options: {'1', '2', '2*R', '2*C', '2*R*C'}.

    Returns:
        list: losses.

    """
    pyro.clear_param_store()
    losses = []
    for _ in range(num_iter):
        loss = svi.step(data, codes, c, r, batch_size, params_mode)
        losses.append(loss)
    return losses


def e_step(data, codes, w, temperature, sigma, c, r, params_mode='2*R*C'):
    """Estimate the posterior probability for spot assignment.

    Args:
        data (torch.array): Input data formatted as torch array with shape ``[num_spots, r * c]``.
        codes (torch.array): Codebook formatted as torch array with shape ``[num_barcodes + 1, r * c]``.
        w (torch.array): Weight parameter for each category with shape ``[num_barcodes + 1, r * c]``.
        temperature (torch.array): Temperature parameter for relaxed bernoulli, shape depends on `params_mode`.
        sigma (torch.array): Sigma parameter for relaxed bernoulli, shape depends on `params_mode`.
        c (int): Number of channels.
        r (int): Number of rounds.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds. valid options: {'1', '2', '2*R', '2*C', '2*R*C'}

    Returns:
        normalized class probability with shape ``[num_spots, num_barcodes + 1]``.
    """
    k = codes.shape[0]  # num_barcodes + 1
    class_logprobs = np.ones((data.shape[0], k))

    if params_mode == '1':  # one param
        scaled_sigma = codes + (-1)**codes * 0.3 * sigma
        aug_temperature = temperature.unsqueeze(-1).repeat(k, 1)
    elif params_mode == '2':  # two params
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
        ind_start, ind_end = idx * \
            batch_sz, torch.min(torch.tensor([(idx+1)*batch_sz, len(data)]))
        for idx in range(k):
            dist = RelaxedBernoulli(
                temperature=aug_temperature[idx], probs=scaled_sigma[idx]).to_event(1)
            class_logprobs[ind_start:ind_end, idx] = (
                w[idx].log() + dist.log_prob(data[ind_start:ind_end])).cpu().numpy()

    # basically doing a stable_softmax here
    numerator = np.exp(
        class_logprobs - np.max(class_logprobs, axis=1)[:, None])
    class_prob_norm = np.divide(numerator, np.sum(numerator, axis=1)[:, None])

    return class_prob_norm


def decoding_function(spots, barcodes, num_iter=500, batch_size=15000, set_seed=1, params_mode='2*R*C'):
    """Main function for the spot decoding.

    Args:
        spots (numpy.array): Input spot intensity array with shape ``[num_spots, c, r]``.
        barcodes (numpy.array): Input codebook array with shape ``[num_barcodes, c, r]``.
        num_iter (int): Number of iterations for training. Defaults to 200.
        batch_size (int): Size of batch for training. Defaults to 1000.
        set_seed (int): Seed for randomness. Defaults to 1.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds. valid options: {'1', '2', '2*R', '2*C', '2*R*C'}.

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

    num_spots, c, r = spots.shape

    data = torch.tensor(spots).float().transpose(
        1, 2).reshape(spots.shape[0], c * r)
    codes = torch.tensor(barcodes).float().transpose(
        1, 2).reshape(barcodes.shape[0], c * r)

    optim = Adam({'lr': 0.085, 'betas': [0.85, 0.99]})
    svi = SVI(model_constrained_tensor, auto_guide_constrained_tensor,
              optim, loss=TraceEnum_ELBO(max_plate_nesting=1))
    pyro.set_rng_seed(set_seed)
    losses = train(svi, num_iter, data, codes, c, r,
                   min(num_spots, batch_size), params_mode)

    w_star = pyro.param('weights').detach()
    temperature_star = pyro.param('temperature').detach()
    sigma_star = pyro.param('sigma').detach()

    class_probs_star = e_step(
        data, codes, w_star, temperature_star, sigma_star, c, r,  params_mode)

    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.FloatTensor")

    torch_params = {'w_star': w_star.cpu(), 'temperature_star': temperature_star.cpu(
    ), 'sigma_star': sigma_star.cpu(), 'losses': losses}
    results = {'class_probs': class_probs_star, 'params': torch_params}

    return results
