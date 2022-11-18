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


def torch_format(numpy_array):
    D = numpy_array.shape[1] * numpy_array.shape[2]
    return torch.tensor(numpy_array).float().transpose(1, 2).reshape(numpy_array.shape[0], D)

def torch_format_1(torch_array):
    D = torch_array.shape[1] * torch_array.shape[2]
    return torch_array.transpose(1, 2).reshape(torch_array.shape[0], D)


def barcodes_01_from_channels(barcodes_1234, C, R):
    K = barcodes_1234.shape[0]
    barcodes_01 = np.ones((K, C, R))
    for b in range(K):
        barcodes_01[b, :, :] = 1 * np.transpose(barcodes_1234[b, :].reshape(R, 1) == np.arange(1, C + 1))
    return barcodes_01


def e_step(data, w, temperature, sigma, N, K, C, R, codes, params_mode='2*R*C'):
    class_logprobs = np.ones((N, K))

    if params_mode == '1': ## one param
        scaled_sigma = codes + (-1)**codes * 0.3 * sigma
        aug_temperature = temperature.unsqueeze(-1).repeat(K,1)
    elif params_mode == '2': ## two params
        aug_sigma = torch.gather(sigma,0, codes.reshape(-1).long()).reshape(codes.shape)
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        aug_temperature = torch.gather(temperature,0, codes.reshape(-1).long()).reshape(codes.shape)
    elif params_mode == '2*R': ## 2*R params
        sigma_temp = sigma.unsqueeze(-1).repeat(1,1,C)
        sigma_temp1 = torch_format_1(sigma_temp)
        aug_sigma = torch.gather(sigma_temp1, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature_temp = temperature.unsqueeze(-1).repeat(1,1,C)
        temperature_temp1 = torch_format_1(temperature_temp)
        aug_temperature = torch.gather(temperature_temp1, 0, codes.long())
    elif params_mode == '2*C': ## 2*C params
        sigma_temp = sigma.unsqueeze(-1).repeat(1,1,R)
        sigma_temp1 = torch_format_1(sigma_temp)
        aug_sigma = torch.gather(sigma_temp1, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature_temp = temperature.unsqueeze(-1).repeat(1,1,R)
        temperature_temp1 = torch_format_1(temperature_temp)
        aug_temperature = torch.gather(temperature_temp1, 0, codes.long())
    elif params_mode == '2*R*C': ## 2*R*C params
        aug_sigma = torch.gather(sigma, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        aug_temperature = torch.gather(temperature, 0, codes.long())
    else:
        assert False, "%s not supported"%params_mode
    
    batch_sz = 50000
    for idx in range(len(data) // batch_sz + 1): 
        ind_start, ind_end = idx*batch_sz, torch.min(torch.tensor([(idx+1)*batch_sz, len(data)]))
        for k in tqdm(range(K)):
            dist = RelaxedBernoulli(temperature=aug_temperature[k], probs=scaled_sigma[k]).to_event(1)
            class_logprobs[ind_start:ind_end, k] = (w[k].log() + dist.log_prob(data[ind_start:ind_end])).cpu().numpy()

    # basically doing a stable_softmax here
    # class_logprobs = class_logprobs.cpu().numpy()
    numerator = np.exp(class_logprobs - np.max(class_logprobs, axis=1)[:, None])
    class_prob_norm = np.divide(numerator, np.sum(numerator, axis=1)[:, None])

    return class_prob_norm


@config_enumerate
def model_constrained_tensor(data, N, D, C, R, K, codes, batch_size=None, weight_initialization=None, params_mode='2*R*C'):
    w = pyro.param('weights', torch.ones(K) / K, constraint=constraints.simplex)

    if params_mode == '1':
        ## one param
        sigma = pyro.param("sigma", torch.ones(torch.Size([1])) * 0.5, constraint=constraints.unit_interval)
        scaled_sigma = codes + (-1)**codes * 0.3 * sigma
        temperature = pyro.param("temperature", torch.ones(torch.Size([1])) * 0.5, constraint=constraints.unit_interval)
        aug_temperature = temperature.unsqueeze(-1).repeat(K,1)
    elif params_mode == '2': ## two param - one for 0-channel, one for 1-channel
        sigma = pyro.param("sigma", torch.ones(torch.Size([2])) * 0.5, constraint=constraints.unit_interval)
        aug_sigma = torch.gather(sigma,0, codes.reshape(-1).long()).reshape(codes.shape)
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature = pyro.param('temperature', torch.ones(torch.Size([2])) * 0.5, constraint=constraints.unit_interval)
        aug_temperature = torch.gather(temperature,0, codes.reshape(-1).long()).reshape(codes.shape)
    elif params_mode == '2*R': ## 2*R params
        sigma = pyro.param("sigma", torch.ones(torch.Size([2, R])) * 0.5, constraint=constraints.unit_interval)
        sigma_temp = sigma.unsqueeze(-1).repeat(1,1,C)
        sigma_temp1 = torch_format_1(sigma_temp)
        aug_sigma = torch.gather(sigma_temp1, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature = pyro.param("temperature", torch.ones(torch.Size([2, R])) * 0.5, constraint=constraints.unit_interval)
        temperature_temp = temperature.unsqueeze(-1).repeat(1,1,C)
        temperature_temp1 = torch_format_1(temperature_temp)
        aug_temperature = torch.gather(temperature_temp1, 0, codes.long())
    elif params_mode == '2*C': ## 2*C params
        sigma = pyro.param("sigma", torch.ones(torch.Size([2, C])) * 0.5, constraint=constraints.unit_interval)
        sigma_temp = sigma.unsqueeze(-1).repeat(1,1,R)
        sigma_temp1 = torch_format_1(sigma_temp)
        aug_sigma = torch.gather(sigma_temp1, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature = pyro.param("temperature", torch.ones(torch.Size([2, C])) * 0.5, constraint=constraints.unit_interval)
        temperature_temp = temperature.unsqueeze(-1).repeat(1,1,R)
        temperature_temp1 = torch_format_1(temperature_temp)
        aug_temperature = torch.gather(temperature_temp1, 0, codes.long())
    elif params_mode == '2*R*C': ## 2*R*C params
        sigma = pyro.param("sigma", torch.ones(torch.Size([2, D])) * 0.5, constraint=constraints.unit_interval)
        aug_sigma = torch.gather(sigma, 0, codes.long())
        scaled_sigma = codes + (-1)**codes * 0.3 * aug_sigma
        temperature = pyro.param("temperature", torch.ones(torch.Size([2, D])) * 0.5, constraint=constraints.unit_interval)
        aug_temperature = torch.gather(temperature, 0, codes.long())
    else:
        assert False, "%s not supported"%params_mode

    with pyro.plate('data', N, batch_size) as batch:
        z = pyro.sample('z', Categorical(w))
        pyro.sample('obs', RelaxedBernoulli(temperature=aug_temperature[z], probs=scaled_sigma[z]).to_event(1), obs=data[batch])

auto_guide_constrained_tensor = AutoDelta(poutine.block(model_constrained_tensor, expose=['weights', 'temperature', 'sigma']))


def train(svi, num_iterations, data, N, D, C, R, K, codes, batch_size, weight_initialization, params_mode):
    pyro.clear_param_store()
    losses = []
    for j in tqdm(range(num_iterations)):
        loss = svi.step(data, N, D, C, R, K, codes, batch_size, weight_initialization, params_mode)
        losses.append(loss)
    return losses


def decoding_function(spots, barcodes_01, num_iter=60, batch_size=15000, set_seed=1, params_mode='2*R*C'):
    # INPUT:
    # spots: a numpy array of dim N x C x R (number of spots x coding channels x rounds);
    # barcodes_01: a numpy array of dim K x C x R (number of barcodes x coding channels x rounds)
    # OUTPUT:
    # 'class_probs': posterior probabilities computed via e-step
    # 'class_ind': indices of different barcode classes (genes / background / infeasible / nan)
    # 'params': estimated model parameters
    # 'norm_const': constants used for normalization of spots prior to model fitting
    # 'params_mode': how many params are allowed - valid options: {'1', '2', '2*R', '2*C', '2*R*C'}

    # if cuda available, runs on gpu
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
            
    N = spots.shape[0]
    if N == 0:
        print('There are no spots to decode.')
        return
    C = spots.shape[1]
    R = spots.shape[2]
    K = barcodes_01.shape[0]
    D = C * R
    
    weight_initialization = None

    data = torch_format(spots)
    codes = torch_format(barcodes_01)

    # include background in codebook
    bkg_ind = codes.shape[0]
    codes = torch.cat((codes, torch.zeros(1, D)))
    
    ind_keep = np.arange(0, N)
    data_norm = data
    
    # model training:
    optim = Adam({'lr': 0.085, 'betas': [0.85, 0.99]})
    svi = SVI(model_constrained_tensor, auto_guide_constrained_tensor, optim, loss=TraceEnum_ELBO(max_plate_nesting=1))
    pyro.set_rng_seed(set_seed)
    losses = train(svi, num_iter, data_norm[ind_keep, :], len(ind_keep), D, C, R, codes.shape[0], codes, min(len(ind_keep), batch_size), weight_initialization, params_mode)
    # collect estimated parameters
    w_star = pyro.param('weights').detach()
    temperature_star = pyro.param('temperature').detach()
    sigma_star = pyro.param('sigma').detach()
    
    w_star_mod = w_star


    class_probs_star = e_step(data_norm, w_star_mod,temperature_star, sigma_star, N, codes.shape[0], C, R,  codes, params_mode)
    class_probs_star_s = class_probs_star
    inf_ind_s = None
        
    class_probs = class_probs_star_s
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.FloatTensor")

    class_ind = {'genes': np.arange(K), 'bkg': bkg_ind, 'inf': inf_ind_s }
    torch_params = {'w_star': w_star_mod.cpu(), 'temperature_star': temperature_star.cpu(), 'sigma_star': sigma_star.cpu(), 'losses': losses}
    norm_const = {}

    return {'class_probs': class_probs, 'class_ind': class_ind, 'params': torch_params, 'norm_const': norm_const}
