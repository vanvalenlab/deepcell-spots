import numpy as np
import torch
import itertools
import pandas as pd
import random
from tqdm import tqdm
import pyro
from pyro.distributions import *
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro import poutine
from pyro.infer.autoguide import AutoDelta

assert pyro.__version__.startswith('1')

# torch.manual_seed(set_seed)
# random.seed(set_seed)
# np.random.seed(set_seed)

"""auxiliary functions required for decoding"""


def torch_format(numpy_array):
    D = numpy_array.shape[1] * numpy_array.shape[2]
    return torch.tensor(numpy_array).float().transpose(1, 2).reshape(numpy_array.shape[0], D)


def barcodes_01_from_channels(barcodes_1234, C, R):
    K = barcodes_1234.shape[0]
    barcodes_01 = np.ones((K, C, R))
    for b in range(K):
        barcodes_01[b, :, :] = 1 * np.transpose(barcodes_1234[b, :].reshape(R, 1) == np.arange(1, C + 1))  # noqa: E501
    return barcodes_01


def kronecker_product(tr, tc):
    tr_height, tr_width = tr.size()
    tc_height, tc_width = tc.size()
    out_height = tr_height * tc_height
    out_width = tr_width * tc_width
    tiled_tc = tc.repeat(tr_height, tr_width)
    expanded_tr = (tr.unsqueeze(2).unsqueeze(3).repeat(1,
                   tc_height, tc_width, 1).view(out_height, out_width))
    return expanded_tr * tiled_tc


def chol_sigma_from_vec(sigma_vec, D):
    L = torch.zeros(D, D)
    L[torch.tril(torch.ones(D, D)) == 1] = sigma_vec
    return torch.mm(L, torch.t(L))


def e_step(data, w, theta, sigma, N, K, print_training_progress):
    class_probs = torch.ones(N, K)
    if print_training_progress:
        for k in tqdm(range(K)):
            dist = MultivariateNormal(theta[k], sigma)
            class_probs[:, k] = w[k] * torch.exp(dist.log_prob(data))
    else:
        for k in range(K):
            dist = MultivariateNormal(theta[k], sigma)
            class_probs[:, k] = w[k] * torch.exp(dist.log_prob(data))

    class_prob_norm = class_probs.div(torch.sum(class_probs, dim=1, keepdim=True))
    # class_prob_norm[torch.isnan(class_prob_norm)] = 0
    return class_prob_norm


def mat_sqrt(A, D):
    try:
        U, S, V = torch.svd(A + 1e-3 * A.mean() * torch.rand(D, D))
    except:  # noqa: E722
        U, S, V = torch.svd(A + 1e-2 * A.mean() * torch.rand(D, D))
    S_sqrt = torch.sqrt(S)
    return torch.mm(torch.mm(U, torch.diag(S_sqrt)), V.t())


def map_states(data, N, D, C, R, K, codes, batch_size=None, temperature=0):
    # avoid conflict with data plate
    inferred_model = infer_discrete(poutine.replay(model_constrained_tensor,
                                                   trace=poutine.trace(auto_guide_constrained_tensor).get_trace(data, N, D, C, R, K, codes)),  # noqa: E501
                                    temperature=temperature, first_available_dim=-2)
    trace = poutine.trace(inferred_model).get_trace(data, N, D, C, R, K, codes)
    return trace.nodes["z"]["value"]


@config_enumerate
def model_constrained_tensor(data, N, D, C, R, K, codes, batch_size=None):
    w = pyro.param('weights', torch.ones(K) / K, constraint=constraints.simplex)

    # using tensor sigma
    sigma_ch_v = pyro.param('sigma_ch_v', torch.eye(C)[np.tril_indices(C, 0)])
    sigma_ch = chol_sigma_from_vec(sigma_ch_v, C)
    sigma_ro_v = pyro.param('sigma_ro_v', torch.eye(D)[np.tril_indices(R, 0)])
    sigma_ro = chol_sigma_from_vec(sigma_ro_v, R)
    sigma = kronecker_product(sigma_ro, sigma_ch)

    # codes_tr_v = pyro.param('codes_tr_v', 3 * torch.ones(1, D), constraint=constraints.positive)
    codes_tr_v = pyro.param('codes_tr_v',
                            3 * torch.ones(1, D),
                            constraint=constraints.greater_than(1.))
    codes_tr_consts_v = pyro.param('codes_tr_consts_v', -1 * torch.ones(1, D))

    theta = torch.matmul(codes * codes_tr_v + codes_tr_consts_v, mat_sqrt(sigma, D))

    with pyro.plate('data', N, batch_size) as batch:
        z = pyro.sample('z', Categorical(w))
        pyro.sample('obs', MultivariateNormal(theta[z], sigma), obs=data[batch])


auto_guide_constrained_tensor = AutoDelta(poutine.block(model_constrained_tensor,
                                          expose=['weights',
                                                  'codes_tr_v',
                                                  'codes_tr_consts_v',
                                                  'sigma_ch_v',
                                                  'sigma_ro_v']))


def train(svi, num_iterations, data, N, D, C, R, K, codes, print_training_progress, batch_size):
    pyro.clear_param_store()
    losses = []
    if print_training_progress:
        for j in tqdm(range(num_iterations)):
            loss = svi.step(data, N, D, C, R, K, codes, batch_size)
            losses.append(loss)
    else:
        for j in range(num_iterations):
            loss = svi.step(data, N, D, C, R, K, codes, batch_size)
            losses.append(loss)
    return losses


# input - output decoding function
def decoding_function(spots, barcodes_01,
                      num_iter=60, batch_size=15000, up_prc_to_remove=99.95,
                      modify_bkg_prior=True,
                      estimate_bkg=True, estimate_additional_barcodes=None,
                      add_remaining_barcodes_prior=0.05,
                      print_training_progress=True, set_seed=1):

    """
    Args:
        spots: a numpy array of dim N x C x R (number of spots x coding channels x rounds)
        barcodes_01: a numpy array of dim K x C x R (number of barcodes x coding channels x rounds)
        modify_bkg_prior (boolean): defaults to True. Should be False when there is a lot of
            background signal (eg pixel-wise decoding, a lot of noisy boundary tiles)
        estimate_bkg (boolean):
        estimate_addition_barcodes: controls adding additional barcodes during parameter estimation
        add_remaining_barcodes_prior: after model is estimated, infeasible barcodes are used in the
            e-step with given prior
        print_training_progresss (boolean):
        set_seed: int
    Returns:
        class_probs: posterior probabilities computed via e-step
        class_ind: indices of different barcode classes (genes / background / infeasible / nan)
        params: estimated model parameters
        norm_const: constants used for normalization of spots prior to model fitting
    """

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

    # normalize spot values
    ind_keep = np.where(np.sum(data.cpu().numpy() < np.percentile(data.cpu().numpy(),
                                                                  up_prc_to_remove, axis=0),
                               axis=1) == D)[0] if up_prc_to_remove < 100 else np.arange(0, N)
    s = torch.tensor(np.percentile(data[ind_keep, :].cpu().numpy(), 60, axis=0))
    max_s = torch.tensor(np.percentile(data[ind_keep, :].cpu().numpy(), 99.9, axis=0))
    min_s = torch.min(data[ind_keep, :], dim=0).values
    log_add = (s ** 2 - max_s * min_s) / (max_s + min_s - 2 * s)
    log_add = torch.max(-torch.min(data[ind_keep, :], dim=0).values + 1e-10, other=log_add.float())
    data_log = torch.log10(data + log_add)
    data_log_mean = data_log[ind_keep, :].mean(dim=0, keepdim=True)
    data_log_std = data_log[ind_keep, :].std(dim=0, keepdim=True)
    data_norm = (data_log - data_log_mean) / data_log_std  # column-wise normalization

    # model training:
    optim = Adam({'lr': 0.085, 'betas': [0.85, 0.99]})
    svi = SVI(model_constrained_tensor,
              auto_guide_constrained_tensor,
              optim,
              loss=TraceEnum_ELBO(max_plate_nesting=1))
    pyro.set_rng_seed(set_seed)
    losses = train(svi,
                   num_iter,
                   data_norm[ind_keep, :],
                   len(ind_keep), D, C, R,
                   codes.shape[0], codes,
                   print_training_progress,
                   min(len(ind_keep), batch_size))
    # collect estimated parameters
    w_star = pyro.param('weights').detach()
    sigma_ch_v_star = pyro.param('sigma_ch_v').detach()
    sigma_ro_v_star = pyro.param('sigma_ro_v').detach()
    sigma_ro_star = chol_sigma_from_vec(sigma_ro_v_star, R)
    sigma_ch_star = chol_sigma_from_vec(sigma_ch_v_star, C)
    sigma_star = kronecker_product(sigma_ro_star, sigma_ch_star)
    codes_tr_v_star = pyro.param('codes_tr_v').detach()
    codes_tr_consts_v_star = pyro.param('codes_tr_consts_v').detach()
    theta_star = torch.matmul(codes * codes_tr_v_star + codes_tr_consts_v_star,
                              mat_sqrt(sigma_star, D))

    # computing class probabilities with appropriate prior probabilities
    if modify_bkg_prior and w_star.shape[0] > K:
        # making sure that the K barcode classes have higher prior in case there are more than
        # K classes
        w_star_mod = torch.cat((w_star[0:K], w_star[0:K].min().repeat(w_star.shape[0] - K)))
        w_star_mod = w_star_mod / w_star_mod.sum()
    else:
        w_star_mod = w_star

    if add_remaining_barcodes_prior > 0:
        # all possible barcodes
        barcodes_1234 = np.array([p for p in itertools.product(np.arange(1, C + 1), repeat=R)])
        # all possible barcodes in the same format as codes
        codes_inf = np.array(torch_format(barcodes_01_from_channels(barcodes_1234, C, R)).cpu())
        # add the bkg code at the beginning
        codes_inf = np.concatenate((np.zeros((1, D)), codes_inf))
        codes_cpu = codes.cpu()
        for b in range(codes_cpu.shape[0]):  # remove already existing codes
            r = np.array(codes_cpu[b, :], dtype=np.int32)
            if np.where(np.all(codes_inf == r, axis=1))[0].shape[0] != 0:
                i = np.reshape(np.where(np.all(codes_inf == r, axis=1)), (1,))[0]
                codes_inf = np.delete(codes_inf, i, axis=0)
        if not estimate_bkg:
            bkg_ind = codes_cpu.shape[0]
            inf_ind = np.append(inf_ind, codes_cpu.shape[0] + 1 + np.arange(codes_inf.shape[0]))
        else:
            inf_ind = np.append(inf_ind, codes_cpu.shape[0] + np.arange(codes_inf.shape[0]))
        codes_inf = torch.tensor(codes_inf).float()
        alpha = (1 - add_remaining_barcodes_prior)
        w_star_all = torch.cat((alpha * w_star_mod,
                                torch.tensor((1 - alpha) / codes_inf.shape[0]).repeat(codes_inf.shape[0])))  # noqa: E501
        class_probs_star = e_step(data_norm, w_star_all,
                                  torch.matmul(torch.cat((codes, codes_inf)) * codes_tr_v_star + codes_tr_consts_v_star.repeat(w_star_all.shape[0], 1),  # noqa: E501
                                               mat_sqrt(sigma_star, D)),
                                  sigma_star,
                                  N,
                                  w_star_all.shape[0],
                                  print_training_progress)
    else:
        class_probs_star = e_step(data_norm,
                                  w_star_mod,
                                  theta_star,
                                  sigma_star,
                                  N,
                                  codes.shape[0],
                                  print_training_progress)

    # collapsing added barcodes
    class_probs_star_s = torch.cat((torch.cat((class_probs_star[:, 0:K],
                                               class_probs_star[:, bkg_ind].reshape((N, 1))), dim=1),  # noqa: E501
                                    torch.sum(class_probs_star[:, inf_ind], dim=1).reshape((N, 1))), dim=1)  # noqa: E501
    inf_ind_s = inf_ind[0]
    # adding another class if there are NaNs
    nan_spot_ind = torch.unique((torch.isnan(class_probs_star_s)).nonzero(as_tuple=False)[:, 0])
    if nan_spot_ind.shape[0] > 0:
        nan_class_ind = class_probs_star_s.shape[1]
        class_probs_star_s = torch.cat((class_probs_star_s, torch.zeros((class_probs_star_s.shape[0], 1))), dim=1)  # noqa: E501
        class_probs_star_s[nan_spot_ind, :] = 0
        class_probs_star_s[nan_spot_ind, nan_class_ind] = 1
    else:
        nan_class_ind = np.empty((0,), dtype=np.int32)

    class_probs = class_probs_star_s.cpu().numpy()
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.FloatTensor")

    class_ind = {'genes': np.arange(K), 'bkg': bkg_ind, 'inf': inf_ind_s, 'nan': nan_class_ind}
    torch_params = {'w_star': w_star_mod.cpu(), 'sigma_star': sigma_star.cpu(),
                    'sigma_ro_star': sigma_ro_star.cpu(), 'sigma_ch_star': sigma_ch_star.cpu(),
                    'theta_star': theta_star.cpu(), 'codes_tr_consts_v_star': codes_tr_consts_v_star.cpu(),  # noqa: E501
                    'codes_tr_v_star': codes_tr_v_star.cpu(), 'losses': losses}
    norm_const = {'log_add': log_add, 'data_log_mean': data_log_mean, 'data_log_std': data_log_std}

    return {'class_probs': class_probs, 'class_ind': class_ind, 'params': torch_params, 'norm_const': norm_const}  # noqa: E501


# function handling output of decoding
def decoding_output_to_dataframe(out, df_class_names, df_class_codes):
    val = out['class_probs'].max(axis=1)
    ind = out['class_probs'].argmax(axis=1)
    K = len(out['class_ind']['genes'])
    decoded = ind + 1
    decoded[np.isin(ind, out['class_ind']['inf'])] = K + 1  # inf class
    decoded[np.isin(ind, out['class_ind']['bkg'])] = K + 2  # bkg class
    decoded[np.isin(ind, out['class_ind']['nan'])] = K + 3  # NaN class
    decoded_spots_df = pd.DataFrame(columns=['Name', 'Code', 'Probability'])
    decoded_spots_df['Name'] = df_class_names[decoded - 1]
    decoded_spots_df['Code'] = df_class_codes[decoded - 1]
    decoded_spots_df['Probability'] = val
    return decoded_spots_df


# function creating a heatmap for plotting spatial patterns
def heatmap_pattern(decoded_df, name, grid=150, thr=0.7, plot_probs=True):
    if 'Probability' not in decoded_df.columns:
        if 'Score' not in decoded_df.columns:
            plot_probs = False
            x_coord = np.floor(decoded_df.X[(decoded_df.Name == name)].to_numpy(dtype=np.double) / grid).astype(np.int32)  # noqa: E501
            y_coord = np.floor(decoded_df.Y[(decoded_df.Name == name)].to_numpy(dtype=np.double) / grid).astype(np.int32)  # noqa: E501
        else:
            x_coord = np.floor(decoded_df.X[(decoded_df.Name == name) & (decoded_df.Score > thr)].to_numpy(dtype=np.double) / grid).astype(np.int32)  # noqa: E501
            y_coord = np.floor(decoded_df.Y[(decoded_df.Name == name) & (decoded_df.Score > thr)].to_numpy(dtype=np.double) / grid).astype(np.int32)  # noqa: E501
    else:
        x_coord = np.floor(decoded_df.X[(decoded_df.Name == name) & (decoded_df.Probability > thr)].to_numpy(dtype=np.double) / grid).astype(np.int32)  # noqa: E501
        y_coord = np.floor(decoded_df.Y[(decoded_df.Name == name) & (decoded_df.Probability > thr)].to_numpy(dtype=np.double) / grid).astype(np.int32)  # noqa: E501
    H = np.zeros((int(np.ceil(decoded_df.Y.to_numpy(dtype=np.double).max() / grid)), int(np.ceil(decoded_df.X.to_numpy(dtype=np.double).max() / grid))))  # noqa: E501
    if plot_probs:
        if 'Probability' in decoded_df.columns:
            prob = decoded_df.Probability[decoded_df.Name == name].to_numpy(dtype=np.double)
        elif 'Score' in decoded_df.columns:
            prob = decoded_df.Score[decoded_df.Name == name].to_numpy(dtype=np.double)
        prob[prob < thr] = 0
        for i in range(len(x_coord)):
            H[y_coord[i], x_coord[i]] = H[y_coord[i], x_coord[i]] + prob[i]
    else:
        coords = np.concatenate((y_coord.reshape((len(x_coord), 1)),
                                 x_coord.reshape((len(x_coord), 1))), axis=1)
        coords_u, coords_c = np.unique(coords, axis=0, return_counts=True)
        H[coords_u[:, 0], coords_u[:, 1]] = coords_c
    return H
