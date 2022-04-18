from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve

def sample_predictions(tau_type_to_files, pt_bin, eta_bin, dm_set, n_per_split, n_splits):
    splits_per_tau_type = defaultdict(dict) # to collect splits per tau type
    key_list = ['add_columns', 'predictions', 'labels'] 
    for tau_type, files in tau_type_to_files.items():
        
        # read into dataframe all the input files of the given tau type 
        df = pd.concat([pd.read_hdf(f, key='add_columns') for f in files], axis=0, ignore_index=True)
        len_df_add = len(df)

        # apply cuts and save indices of objects which passed the selection
        df = df.query(f'tau_pt>{pt_bin[0]} and tau_pt<{pt_bin[1]}')
        df = df.query(f'abs(tau_eta)>{eta_bin[0]} and abs(tau_eta)<{eta_bin[1]}')
        df = df.query(f'tau_decayMode in {dm_set} ')
        df = df.sample(frac=1) # shuffle to mix various sources for the given tau type
        select_idx = df.index
        df = df.reset_index(drop=True)
        print(f'\n-> Selected in total ({len(df)}) objects of tau type ({tau_type})')
        print(f'   Will split it into ({n_splits}) splits with ({n_per_split}) objects per split')

        if len(df) < n_splits*n_per_split:
            raise RuntimeError(f'Selected number of tau candidates ({len(df)}) is less then requested number ({n_splits*n_per_split})')
        df_splits = np.array_split(df, len(df)//n_per_split)[:n_splits]
        splits_per_tau_type[tau_type]['add_columns'] = df_splits 

        for k in [k_ for k_ in key_list if k_!='add_columns']: # loop over the other key types except for "add_columns"
            df = pd.concat([pd.read_hdf(f, key=k) for f in files], axis=0, ignore_index=True)
            assert len(df)==len_df_add
            df = df.loc[select_idx].reset_index(drop=True) # select objects            
            df_splits = np.array_split(df, len(df)//n_per_split)[:n_splits]
            splits_per_tau_type[tau_type][k] = df_splits

    print()
    samples = []
    for i in range(n_splits): 
        splits = {}
        for k in key_list: # concat across tau_types per split
            splits[k] = pd.concat([splits_per_tau_type[tau_type][k][i] for tau_type in splits_per_tau_type.keys()], axis=0, ignore_index=True)
        samples.append(splits)

    return samples

def derive_roc_curves(prediction_samples, tpr_grid, deeptau_score_name):
    fpr_arrays, fpr_deeptau_arrays = [], []
    auc, auc_deeptau = [], []

    for pred_data in prediction_samples:
        fpr, tpr, _ = roc_curve(pred_data['labels']['label_tau'].values, pred_data['predictions']['pred_tau'].values, pos_label=1) 
        interpolator = interpolate.interp1d(tpr, fpr)
        fpr_arrays.append(interpolator(tpr_grid))

        fpr_deeptau, tpr_deeptau, _ = roc_curve(pred_data['labels']['label_tau'].values, pred_data['add_columns'][deeptau_score_name].values, pos_label=1)
        interpolator_deeptau = interpolate.interp1d(tpr_deeptau, fpr_deeptau)
        fpr_deeptau_arrays.append(interpolator_deeptau(tpr_grid))

        auc.append(roc_auc_score(pred_data['labels']['label_tau'], pred_data['predictions']['pred_tau']))
        auc_deeptau.append(roc_auc_score(pred_data['labels']['label_tau'], pred_data['add_columns'][deeptau_score_name]))

    fpr_mean = np.mean(fpr_arrays, axis=0)
    fpr_std = np.std(fpr_arrays, axis=0)
    fpr_deeptau_mean = np.mean(fpr_deeptau_arrays, axis=0)
    fpr_deeptau_std = np.std(fpr_deeptau_arrays, axis=0)

    auc_mean = np.mean(auc, axis=0)
    auc_std = np.std(auc, axis=0)
    auc_deeptau_mean = np.mean(auc_deeptau, axis=0)
    auc_deeptau_std = np.std(auc_deeptau, axis=0)

    return (fpr_mean, fpr_std, auc_mean, auc_std), (fpr_deeptau_mean, fpr_deeptau_std, auc_deeptau_mean, auc_deeptau_std)