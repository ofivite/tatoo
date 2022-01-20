import os
import time
import shutil
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from utils.data_preprocessing import get_tau_arrays, preprocess_taus
import tensorflow as tf

@hydra.main(config_path='configs', config_name='create_dataset')
def main(cfg: DictConfig) -> None:

    print('\n-> Retrieving input awkward arrays')
    time_0 = time.time()
    a = get_tau_arrays(cfg.data_cfg)
    time_1 = time.time()
    print(f'   took: {(time_1-time_0):.1f} s.')

    print('\n-> Preprocessing')
    (X_train, y_train), (X_val, y_val) = preprocess_taus(a, cfg.vs_type, cfg.feature_names, cfg.n_samples_train, cfg.n_samples_val)
    time_2 = time.time()
    print(f'   took: {(time_2-time_1):.1f} s.') 

    print('\n-> Preparing TF datasets')
    # create train data set
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_data = train_data.cache()
    train_data = train_data.shuffle(cfg.shuffle_buffer_size).batch(cfg.train_batch_size)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

    # create validation data set
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_data = val_data.cache()
    val_data = val_data.batch(cfg.val_batch_size)
    val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)
    time_3 = time.time()
    print(f'took: {(time_3-time_2):.1f} s.') 

    # remove existing datasets
    path_to_train_dataset = to_absolute_path(f'datasets/{cfg.dataset_name}/train/{cfg.vs_type}')
    path_to_val_dataset = to_absolute_path(f'datasets/{cfg.dataset_name}/val/{cfg.vs_type}')
    if os.path.exists(path_to_train_dataset):
        shutil.rmtree(path_to_train_dataset)
    if os.path.exists(path_to_val_dataset):
        shutil.rmtree(path_to_val_dataset)

    # save
    print('\n-> Saving TF datasets')
    tf.data.experimental.save(train_data, path_to_train_dataset)
    tf.data.experimental.save(val_data, path_to_val_dataset)
    time_4 = time.time()
    print(f'took: {time_4 - time_3}') 
    print(f'total time: {(time_4-time_0):.1f} s.\n') 

if __name__ == '__main__':
    main()