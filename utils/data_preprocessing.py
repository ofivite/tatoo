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

def get_tau_arrays(data_cfg):
    taus = []
    for sample, tau_types in data_cfg['input_samples'].items():
        target_selection = ' | '.join([f'(tauType=={data_cfg["target_map"][tau_type]})' for tau_type in tau_types]) # select only taus of required classes
        print(f'      - {sample}')
        
        # open ROOT file and retireve awkward arrays
        with uproot.open(to_absolute_path(f'{sample}.root')) as f:
            a = f[data_cfg['tree_name']].arrays(data_cfg['input_branches'], cut=target_selection, how='zip')
                
            # add target labels
            for tau_type in tau_types:
                a[f'node_{tau_type}'] = ak.values_astype(a['tauType'] == data_cfg['target_map'][tau_type], np.int32)
                n_samples = np.sum(a[f'node_{tau_type}'])
                print(f'          {tau_type}: {n_samples} samples')

        # append to array list 
        taus.append(a)
    
    # concat all samples together and shuffle
    taus = ak.concatenate(taus, axis=0)
    taus = taus[np.random.permutation(len(taus))]

    return taus

def awkward_to_ragged(a, feature_names):
    pf_lengths = ak.count(a['pfCand', feature_names[0]], axis=1)
    ragged_pf_features = []
    for feature in feature_names:
        pf_a = ak.flatten(a['pfCand', feature])
        pf_a = ak.values_astype(pf_a, np.float32)
        ragged_pf_features.append(tf.RaggedTensor.from_row_lengths(pf_a, pf_lengths))
    ragged_pf = tf.stack(ragged_pf_features, axis=-1)
    return ragged_pf

def preprocess_taus(a, vs_type, feature_names, n_samples_train, n_samples_val):
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

    # split targets
    targets_train = ak.to_pandas(a_train[['node_tau', f'node_{vs_type}']])
    targets_val = ak.to_pandas(a_val[['node_tau', f'node_{vs_type}']])
    print(targets_train.value_counts())
    print(targets_val.value_counts())
    targets_train = targets_train.values
    targets_val = targets_val.values

    # convert to ragged arrays with only required features
    X_train = awkward_to_ragged(a_train, feature_names)
    X_val = awkward_to_ragged(a_val, feature_names)

    return (X_train, targets_train), (X_val, targets_val)