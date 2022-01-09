import uproot
import awkward as ak
import tensorflow as tf
import h5py
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path

def get_tau_targets(data_sample, file_name):
    with h5py.File(to_absolute_path(f'data/{data_sample}/{file_name}_pred.h5'), "r") as f:
        target_columns = [i.decode("utf-8") for i in f.get('targets/block0_items')]
        targets = np.array(f.get('targets/block0_values'), dtype=np.int32)
    return pd.DataFrame(targets, columns=target_columns)

def get_tau_arrays(datasets, tree_name):
    # retrieve tau consituents
    arrays = []
    for data_sample, file_name in datasets.items():
        with uproot.open(to_absolute_path(f'data/{data_sample}/{file_name}.root')) as f:
            a = f[tree_name].arrays(['pfCand_pt', 'pfCand_eta', 'pfCand_phi', 'pfCand_particleType',
                        'tau_pt', 'tau_eta', 'tau_phi', 'genLepton_kind',
                        'tau_decayMode', 'tau_decayModeFinding',], how='zip')
        # add target labels
        targets = get_tau_targets(data_sample, file_name)
        for c in targets.columns: 
            a[c] = targets[c]
        arrays.append(a)
    return ak.concatenate(arrays, axis=0)

def preprocess_taus(a, vs_type, n_samples_train, n_samples_val):
    # remove taus with abnormal phi
    a = a[np.abs(a['tau_phi'])<2.*np.pi] 

    # shift delta phi into [-pi, pi] range 
    dphi_array = (a['pfCand', 'phi'] - a['tau_phi'])
    dphi_array = np.where(dphi_array <= np.pi, dphi_array, dphi_array - 2*np.pi)
    dphi_array = np.where(dphi_array >= -np.pi, dphi_array, dphi_array + 2*np.pi)

    # add features
    a['pfCand', 'dphi'] = dphi_array
    a['pfCand', 'deta'] = a['pfCand', 'eta'] - a['tau_eta']
    a['pfCand', 'rel_pt'] = a['pfCand', 'pt'] / a['tau_pt']
    a['pfCand', 'r'] = np.sqrt(np.square(a['pfCand', 'deta']) + np.square(a['pfCand', 'dphi']))
    a['pfCand', 'theta'] = np.arctan2(a['pfCand', 'dphi'], a['pfCand', 'deta']) # dphi -> y, deta -> x
    a['pfCand', 'particle_type'] = a['pfCand', 'particleType'] - 1

    # select classes 
    a_taus = a[a['node_tau'] == 1]
    a_vs_type = a[a[f'node_{vs_type}'] == 1]

    # concat classes & shuffle
    a_train = ak.concatenate([a_taus[:n_samples_train], a_vs_type[:n_samples_train]], axis=0)
    a_val = ak.concatenate([a_taus[n_samples_train:n_samples_train+n_samples_val], \
                            a_vs_type[n_samples_train:n_samples_train+n_samples_val]], axis=0)
    a_train = a_train[np.random.permutation(len(a_train))]
    a_val = a_val[np.random.permutation(len(a_val))]

    return a_train, a_val

def awkward_to_ragged(a, feature_names):
    pf_lengths = ak.count(a['pfCand', 'pt'], axis=1)
    ragged_pf_features = []
    for feature in feature_names:
        pf_a = ak.flatten(a['pfCand', feature])
        pf_a = ak.values_astype(pf_a, np.float32)
        ragged_pf_features.append(tf.RaggedTensor.from_row_lengths(pf_a, pf_lengths))
    ragged_pf = tf.stack(ragged_pf_features, axis=-1)
    return ragged_pf