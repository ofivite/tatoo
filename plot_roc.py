import os
from glob import glob
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve

@hydra.main(config_path='configs', config_name='plot_roc')
def main(cfg: DictConfig) -> None:
    path_to_artifacts = to_absolute_path(f'{cfg["path_to_mlflow"]}/{cfg["experiment_id"]}/{cfg["run_id"]}/artifacts')
    vs_type = cfg["vs_type"]

    figure_filename = f'{cfg["figure_filename"]}.pdf'
    fig, ax = plt.subplots(figsize=(10, 10))
    lw = 3

    tpr_grid = np.linspace(start=0., stop=1., num=100000, endpoint=True)
    fpr_arrays, fpr_deeptau_arrays = [], []
    auc, auc_deeptau = [], []

    for p in glob(f'{path_to_artifacts}/predictions/{cfg["dataset_name"]}/*/{vs_type}/*.h5'):
        predictions = pd.read_hdf(p, key='predictions', start=0, stop=cfg["n_samples_to_take"]) 
        labels = pd.read_hdf(p, key='labels', start=0, stop=cfg["n_samples_to_take"])
        deeptau_scores = pd.read_hdf(p, key='deeptau_scores', start=0, stop=cfg["n_samples_to_take"])

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

    plt.plot(
        tpr_grid,
        fpr_deeptau_mean,
        color="gray",
        lw=lw,
        # linestyle='dashed',
        label=f'DeepTau, AUC={auc_deeptau_mean:.{cfg["auc_precision"][vs_type]}f} $\pm$ {auc_deeptau_std:.{cfg["auc_precision"][vs_type]}f}',
    )
    plt.plot(
        tpr_grid,
        fpr_mean,
        color="brown",
        lw=lw,
        # linestyle='dashed',
        label=f'{cfg["model_name"]}, AUC={auc_mean:.{cfg["auc_precision"][vs_type]}f} $\pm$ {auc_std:.{cfg["auc_precision"][vs_type]}f}',
    )
    plt.fill_between(
        tpr_grid,
        fpr_mean-fpr_std,
        fpr_mean+fpr_std,
        color="brown",
        alpha=0.12
    )
    plt.fill_between(
        tpr_grid,
        fpr_deeptau_mean-fpr_deeptau_std,
        fpr_deeptau_mean+fpr_deeptau_std,
        color="gray",
        alpha=0.12
    )

    # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.yscale('log')
    plt.xlim(cfg["xlim"][vs_type])
    plt.ylim(cfg["ylim"][vs_type])
    plt.xlabel("Tau efficiency", fontsize=20)
    plt.ylabel("Background mis-id", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.title(f'vs ({vs_type}), dataset ({cfg["dataset_name"]}, {cfg["n_samples_to_take"] // 1000}k samples)', fontsize=24)
    plt.grid()
    plt.legend(loc="upper left", fontsize=22)
    plt.show()
    fig.savefig(figure_filename)

    # log to mlflow and delete intermediate file
    mlflow.set_tracking_uri(f'file://{to_absolute_path(cfg["path_to_mlflow"])}')
    with mlflow.start_run(experiment_id=cfg["experiment_id"], run_id=cfg["run_id"]) as active_run:
        mlflow.log_artifact(figure_filename, 'plots')
    os.remove(figure_filename)

if __name__ == '__main__':
    main()