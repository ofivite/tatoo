import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve

def derive_roc_curve(predictions, n_samples_to_take, tpr_grid):
    fpr_arrays, fpr_deeptau_arrays = [], []
    auc, auc_deeptau = [], []

    for p in predictions:
        predictions = pd.read_hdf(p, key='predictions', start=0, stop=n_samples_to_take) 
        labels = pd.read_hdf(p, key='labels', start=0, stop=n_samples_to_take)
        deeptau_scores = pd.read_hdf(p, key='deeptau_scores', start=0, stop=n_samples_to_take)

        fpr, tpr, _ = roc_curve(labels['label_tau'].values, predictions['pred_tau'].values, pos_label=1) 
        interpolator = interpolate.interp1d(tpr, fpr)
        fpr_arrays.append(interpolator(tpr_grid))

        fpr_deeptau, tpr_deeptau, _ = roc_curve(labels['label_tau'].values, deeptau_scores['deeptau_score'].values, pos_label=1)
        interpolator_deeptau = interpolate.interp1d(tpr_deeptau, fpr_deeptau)
        fpr_deeptau_arrays.append(interpolator_deeptau(tpr_grid))

        auc.append(roc_auc_score(labels['label_tau'], predictions['pred_tau']))
        auc_deeptau.append(roc_auc_score(labels['label_tau'], deeptau_scores['deeptau_score']))

    fpr_mean = np.mean(fpr_arrays, axis=0)
    fpr_std = np.std(fpr_arrays, axis=0)
    fpr_deeptau_mean = np.mean(fpr_deeptau_arrays, axis=0)
    fpr_deeptau_std = np.std(fpr_deeptau_arrays, axis=0)

    auc_mean = np.mean(auc, axis=0)
    auc_std = np.std(auc, axis=0)
    auc_deeptau_mean = np.mean(auc_deeptau, axis=0)
    auc_deeptau_std = np.std(auc_deeptau, axis=0)

    return (fpr_mean, fpr_std, auc_mean, auc_std), (fpr_deeptau_mean, fpr_deeptau_std, auc_deeptau_mean, auc_deeptau_std)