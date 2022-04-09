import os
from glob import glob
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

import mlflow
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_virtual_device_configuration(physical_devices[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10*1024)])

@hydra.main(config_path='configs', config_name='predict')
def main(cfg: DictConfig) -> None:

    print('\n-> Loading model\n')
    if cfg["checkpoint"] is not None:
        path_to_model = to_absolute_path(f'{cfg["path_to_mlflow"]}/{cfg["experiment_id"]}/{cfg["run_id"]}/artifacts/checkpoints/{cfg["checkpoint"]}')
    else:
        path_to_model = to_absolute_path(f'{cfg["path_to_mlflow"]}/{cfg["experiment_id"]}/{cfg["run_id"]}/artifacts/model/')
    model = load_model(path_to_model)

    for p in glob(to_absolute_path(f'datasets/{cfg["dataset_name"]}/{cfg["dataset_type"]}/{cfg["filename_prefix"]}*/{cfg["vs_type"]}')):
        file_name = p.split('/')[-2]
        dataset = tf.data.experimental.load(p)

        print(f'\n-> Predicting {file_name}')
        predictions, labels, deeptau_scores = [], [], []
        for (X, y, deeptau_score) in dataset:
            predictions.append(model.predict(X))
            labels.append(y)
            deeptau_scores.append(deeptau_score)

        predictions = tf.concat(predictions, axis=0).numpy()
        labels = tf.concat(labels, axis=0).numpy()
        deeptau_scores = tf.concat(deeptau_scores, axis=0).numpy()

        print(f'   Saving to hdf5\n')
        predictions = pd.DataFrame({f'pred_{tau_type}': predictions[:, int(idx)] for tau_type, idx in cfg["tau_type_to_node"].items()})
        labels = pd.DataFrame({f'label_{tau_type}': labels[:, int(idx)] for tau_type, idx in cfg["tau_type_to_node"].items()}, dtype=np.int64)
        deeptau_scores = pd.DataFrame({f'deeptau_score': deeptau_scores})
        
        predictions.to_hdf(f'{cfg["output_filename"]}.h5', key='predictions', mode='w', format='fixed', complevel=1, complib='zlib')
        labels.to_hdf(f'{cfg["output_filename"]}.h5', key='labels', mode='r+', format='fixed', complevel=1, complib='zlib')
        deeptau_scores.to_hdf(f'{cfg["output_filename"]}.h5', key='deeptau_scores', mode='r+', format='fixed', complevel=1, complib='zlib')
        
        # log to mlflow and delete intermediate file
        mlflow.set_tracking_uri(f'file://{to_absolute_path(cfg["path_to_mlflow"])}')
        with mlflow.start_run(experiment_id=cfg["experiment_id"], run_id=cfg["run_id"]) as active_run:
            mlflow.log_artifact(f'{cfg["output_filename"]}.h5', f'predictions/{cfg["dataset_name"]}/{file_name}/{cfg["vs_type"]}')
        os.remove(f'{cfg["output_filename"]}.h5')

if __name__ == '__main__':
    main()