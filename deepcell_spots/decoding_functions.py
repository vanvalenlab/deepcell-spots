import numpy as np
import torch
import itertools
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import pyro
from pyro.distributions import RelaxedBernoulli, Categorical, constraints
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
import gc

assert pyro.__version__.startswith('1')


# auxiliary functions required for decoding
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

def e_step(data, w, temperature, sigma, N, K, C, R, codes, print_training_progress, params_mode='2*R*C'):
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
    
    ## original implementation - 265*20 params
    # sigma = pyro.param("sigma", torch.ones_like(codes) * 0.5, constraint=constraints.unit_interval)
    # scaled_sigma = codes + (-1)**codes * 0.3 * sigma

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


def train(svi, num_iterations, data, N, D, C, R, K, codes, print_training_progress, batch_size, weight_initialization, params_mode):
    pyro.clear_param_store()
    losses = []
    if print_training_progress:
        for j in tqdm(range(num_iterations)):
            loss = svi.step(data, N, D, C, R, K, codes, batch_size, weight_initialization, params_mode)
            losses.append(loss)
    else:
        for j in range(num_iterations):
            loss = svi.step(data, N, D, C, R, K, codes, batch_size, weight_initialization, params_mode)
            losses.append(loss)
    return losses


# input - output decoding function
def decoding_function(spots, barcodes_01,
                      num_iter=60, batch_size=15000, up_prc_to_remove=99.95,
                      modify_bkg_prior=False, # should be False when there is a lot of background signal (eg pixel-wise decoding, a lot of noisy boundary tiles)
                      estimate_bkg=True, estimate_additional_barcodes=None, # controls adding additional barcodes during parameter estimation
                      add_remaining_barcodes_prior=0, # after model is estimated, infeasible barcodes are used in the e-step with given prior
                      print_training_progress=True, set_seed=1, normalization=False, 
                      params_mode='2*R*C'):
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

    # include background / any additional barcode in codebook
    if estimate_bkg:
        bkg_ind = codes.shape[0]
        codes = torch.cat((codes, torch.zeros(1, D)))
    else:
        bkg_ind = np.empty((0,), dtype=np.int32)
    if np.any(estimate_additional_barcodes is not None):
        inf_ind = codes.shape[0] + np.arange(estimate_additional_barcodes.shape[0])
        codes = torch.cat((codes, torch_format(estimate_additional_barcodes)))
    else:
        inf_ind = np.empty((0,), dtype=np.int32)

    ind_keep = np.arange(0, N)
    data_norm = data
    
    # model training:
    optim = Adam({'lr': 0.085, 'betas': [0.85, 0.99]})
    svi = SVI(model_constrained_tensor, auto_guide_constrained_tensor, optim, loss=TraceEnum_ELBO(max_plate_nesting=1))
    pyro.set_rng_seed(set_seed)
    losses = train(svi, num_iter, data_norm[ind_keep, :], len(ind_keep), D, C, R, codes.shape[0], codes, print_training_progress, min(len(ind_keep), batch_size), weight_initialization, params_mode)
    # collect estimated parameters
    w_star = pyro.param('weights').detach()
    temperature_star = pyro.param('temperature').detach()
    sigma_star = pyro.param('sigma').detach()
    
    # computing class probabilities with appropriate prior probabilities
    if modify_bkg_prior and w_star.shape[0] > K:
        # making sure that the K barcode classes have higher prior in case there are more than K classes
        w_star_mod = torch.cat((w_star[0:K], w_star[0:K].min().repeat(w_star.shape[0] - K)))
        w_star_mod = w_star_mod / w_star_mod.sum()
    else:
        w_star_mod = w_star

    if add_remaining_barcodes_prior > 0:
        pass
        # barcodes_1234 = np.array([p for p in itertools.product(np.arange(1, C + 1), repeat=R)])  # all possible barcodes
        # codes_inf = np.array(torch_format(barcodes_01_from_channels(barcodes_1234, C, R)).cpu())  # all possible barcodes in the same format as codes
        # codes_inf = np.concatenate((np.zeros((1, D)), codes_inf))  # add the bkg code at the beginning
        # codes_cpu = codes.cpu()
        # for b in range(codes_cpu.shape[0]):  # remove already existing codes
        #     r = np.array(codes_cpu[b, :], dtype=np.int32)
        #     if np.where(np.all(codes_inf == r, axis=1))[0].shape[0]!=0:
        #         i = np.reshape(np.where(np.all(codes_inf == r, axis=1)), (1,))[0]
        #         codes_inf = np.delete(codes_inf, i, axis=0)
        # if not estimate_bkg:
        #     bkg_ind = codes_cpu.shape[0]
        #     inf_ind = np.append(inf_ind, codes_cpu.shape[0] + 1 + np.arange(codes_inf.shape[0]))
        # else:
        #     inf_ind = np.append(inf_ind, codes_cpu.shape[0] + np.arange(codes_inf.shape[0]))
        # codes_inf = torch.tensor(codes_inf).float()
        # alpha = (1 - add_remaining_barcodes_prior)
        # w_star_all = torch.cat((alpha * w_star_mod, torch.tensor((1 - alpha) / codes_inf.shape[0]).repeat(codes_inf.shape[0])))
        # # theta_star_temp = torch.cat((codes, codes_inf)) * codes_tr_v_star + codes_tr_consts_v_star.repeat(w_star_all.shape[0], 1)
        # sigma_star_temp = torch.cat((sigma_star, torch.tile(torch.mean(sigma_star, axis=0), (w_star_all.shape[0]-K-1, 1))), 0)
        # class_probs_star = e_step(data_norm, w_star_all,temperature_star,
        #                           sigma_star_temp, N, w_star_all.shape[0], C, R, 
        #                           torch.cat((codes, codes_inf)),
        #                           print_training_progress, params_mode)
        # # collapsing added barcodes
        # class_probs_star_s = torch.cat((torch.cat((class_probs_star[:, 0:K], class_probs_star[:, bkg_ind].reshape((N, 1))), dim=1), torch.sum(class_probs_star[:, inf_ind], dim=1).reshape((N, 1))), dim=1)
        # inf_ind_s = inf_ind[0]
    else:
        class_probs_star = e_step(data_norm, w_star_mod,temperature_star, sigma_star, N, codes.shape[0], C, R,  codes, print_training_progress, params_mode)
        class_probs_star_s = class_probs_star
        inf_ind_s = None


#     # adding another class if there are NaNs
#     nan_spot_ind = torch.unique((torch.isnan(class_probs_star_s)).nonzero(as_tuple=False)[:, 0])
#     if nan_spot_ind.shape[0] > 0:
#         nan_class_ind = class_probs_star_s.shape[1]
#         class_probs_star_s = torch.cat((class_probs_star_s, torch.zeros((class_probs_star_s.shape[0], 1))), dim=1)
#         class_probs_star_s[nan_spot_ind, :] = 0
#         class_probs_star_s[nan_spot_ind, nan_class_ind] = 1
#     else:
#         nan_class_ind = np.empty((0,), dtype=np.int32)
        
    class_probs = class_probs_star_s # .cpu().numpy()
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.FloatTensor")

    class_ind = {'genes': np.arange(K), 'bkg': bkg_ind, 'inf': inf_ind_s } #, 'nan': nan_class_ind
    torch_params = {'w_star': w_star_mod.cpu(), 'temperature_star': temperature_star.cpu(), 'sigma_star': sigma_star.cpu(),
                    # 'sigma_ro_star': sigma_ro_star.cpu(), 'sigma_ch_star': sigma_ch_star.cpu(),
                    # 'theta_star': theta_star.cpu(), 'codes_tr_consts_v_star': codes_tr_consts_v_star.cpu(),
                    # 'codes_tr_v_star': codes_tr_v_star.cpu(), 
                    'losses': losses}
    norm_const = {}

    return {'class_probs': class_probs, 'class_ind': class_ind, 'params': torch_params, 'norm_const': norm_const}
