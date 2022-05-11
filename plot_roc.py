import os
from collections import defaultdict
from glob import glob
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.plotting import derive_roc_curves, sample_predictions

@hydra.main(config_path='configs', config_name='plot_roc')
def main(cfg: DictConfig) -> None:
    vs_type = cfg["vs_type"]
    figure_filename = f'{cfg["figure_filename"]}.pdf'
    fig, ax = plt.subplots(figsize=(10, 10))
    lw = 3
    tpr_grid = np.linspace(start=0., stop=1., num=100000, endpoint=True)

    for model_name, (experiment_id, run_id, linestyle, alpha) in cfg["models"].items():
        path_to_artifacts = to_absolute_path(f'{cfg["path_to_mlflow"]}/{experiment_id}/{run_id}/artifacts')

        tau_type_to_files = defaultdict(list)
        for dataset, tau_types in cfg["datasets"].items():
            for tau_type in tau_types:
                tau_type_to_files[tau_type] += glob(f'{path_to_artifacts}/predictions/{dataset}/*/{tau_type}/*.h5')
        predictions = sample_predictions(tau_type_to_files, cfg["pt_bin"], cfg["eta_bin"], cfg["dm_set"], cfg["n_per_split"], cfg["n_splits"]) 


        (fpr_mean, fpr_std, auc_mean, auc_std), (fpr_deeptau_mean, fpr_deeptau_std, auc_deeptau_mean, auc_deeptau_std) = derive_roc_curves(predictions, tpr_grid, cfg["deeptau_score_name"])
        
        plt.plot(
            tpr_grid,
            fpr_mean,
            color="brown",
            lw=lw,
            linestyle=linestyle,
            alpha=alpha,
            label=f'{model_name}, AUC={auc_mean:.{cfg["auc_precision"][vs_type]}f} $\pm$ {auc_std:.{cfg["auc_precision"][vs_type]}f}',
        )
        plt.fill_between(
            tpr_grid,
            fpr_mean-fpr_std,
            fpr_mean+fpr_std,
            color="brown",
            alpha=0.15
        )

    # will plot DeepTau roc curve taken from the last model's predictions
    plt.plot(
        tpr_grid,
        fpr_deeptau_mean,
        color="gray",
        lw=lw,
        # linestyle='dashed',
        label=f'DeepTau, AUC={auc_deeptau_mean:.{cfg["auc_precision"][vs_type]}f} $\pm$ {auc_deeptau_std:.{cfg["auc_precision"][vs_type]}f}',
    )
    plt.fill_between(
        tpr_grid,
        fpr_deeptau_mean-fpr_deeptau_std,
        fpr_deeptau_mean+fpr_deeptau_std,
        color="gray",
        alpha=0.15
    )

    pt_text = r"$p_T \in ({}, {})$ GeV".format(cfg["pt_bin"][0], cfg["pt_bin"][1])
    eta_text = r"${} < |\eta| < {}$".format(cfg["eta_bin"][0], cfg["eta_bin"][1])
    dm_text = r"DM$ \in {}$".format(cfg["dm_set"])
    ax.text(0.03, 0.8 - len(cfg["models"]) * 0.07, pt_text, fontsize=20, transform=ax.transAxes)
    ax.text(0.03, 0.74 - len(cfg["models"]) * 0.07, eta_text, fontsize=20, transform=ax.transAxes)
    ax.text(0.03, 0.68 - len(cfg["models"]) * 0.07, dm_text, fontsize=20, transform=ax.transAxes)

    # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.yscale('log')
    plt.xlim(cfg["xlim"][vs_type])
    plt.ylim(cfg["ylim"][vs_type])
    plt.xlabel("Tau efficiency", fontsize=20)
    plt.ylabel("Background mis-id", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.title(f'vs ({vs_type}), dataset ({cfg["datasets_mix_name"]}, 2x{cfg["n_per_split"] // 1000}k samples)', fontsize=24)
    plt.grid()
    plt.legend(loc="upper left", fontsize=20)
    plt.show()
    fig.savefig(figure_filename)

    # log to mlflow and delete intermediate file
    mlflow.set_tracking_uri(f'file://{to_absolute_path(cfg["path_to_mlflow"])}')
    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as active_run:
        mlflow.log_artifact(figure_filename, 'plots')
    os.remove(figure_filename)

if __name__ == '__main__':
    main()