from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve

def sample_predictions(tau_type_to_files, n_per_split, n_splits):
    samples = defaultdict(list)
    key_list = ['predictions', 'labels', 'add_columns']
    for k in key_list:
        splits_per_tau_type = [] # to collect splits per tau type
        for tau_type, files in tau_type_to_files.items():
            df = pd.concat([pd.read_hdf(f, key=k) for f in files], axis=0, ignore_index=True)
            df = df.sample(frac=1).reset_index(drop=True) # shuffle before splitting to mix input sources
            df_splits = np.array_split(df, len(df)//n_per_split)[:n_splits]
            splits_per_tau_type.append(df_splits)

        for i in range(n_splits):
            splits = [s[i] for s in splits_per_tau_type]
            splits = pd.concat(splits, axis=0, ignore_index=True)
            samples[k].append(splits)

    samples = [{k: samples[k][i] for k in key_list} for i in range(n_splits)]
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