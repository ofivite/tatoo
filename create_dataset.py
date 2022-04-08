import os
import time
import shutil
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from utils.data_preprocessing import load_from_file, preprocess_array, preprocess_labels, awkward_to_ragged

import tensorflow as tf
import awkward as ak
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_virtual_device_configuration(physical_devices[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8*1024)])

@hydra.main(config_path='configs', config_name='create_dataset')
def main(cfg: DictConfig) -> None:
    time_start = time.time()

    # read from cfg
    tau_type_map = cfg['data_cfg']['tau_type_map']
    tree_name = cfg['data_cfg']['tree_name']
    input_branches = cfg['data_cfg']['input_branches']
    vs_type = cfg['vs_type']
    
    for dataset_type in cfg['data_cfg']['input_files'].keys():
        files = OmegaConf.to_object(cfg['data_cfg']['input_files'][dataset_type])
        print(f'\n-> Processing input files ({dataset_type})')
        n_samples = {'tau': 0, vs_type: 0}

        for file_name, tau_types in files.items():
            time_0 = time.time()

            # open ROOT file, read awkward array
            a = load_from_file(file_name, tree_name, input_branches)
            time_1 = time.time()
            print(f'        Loading: took {(time_1-time_0):.1f} s.')

            # preprocess awkward array & add labels
            a = preprocess_array(a)
            a = preprocess_labels(a, tau_types, tau_type_map)
            time_2 = time.time()
            print(f'        Preprocessing: took {(time_2-time_1):.1f} s.')

            # convert awkward to TF ragged arrays
            X = awkward_to_ragged(a, cfg['feature_names']) # keep only feats from feature_names
            y = ak.to_pandas(a[['node_tau', f'node_{vs_type}']]).values
            for k in n_samples.keys():
                n_samples[k] += np.sum(a[f'node_{k}'])
            if cfg['return_deeptau_score']:
                deeptau_score = ak.to_pandas(a[f'tau_byDeepTau2017v2p1VS{vs_type}raw'])
                deeptau_score = np.squeeze(deeptau_score.values)
                data = (X, y, deeptau_score)
            else:
                data = (X, y)
            
            # create TF dataset 
            dataset = tf.data.Dataset.from_tensor_slices(data)
            if cfg['cache']:
                dataset = dataset.cache()
            dataset = dataset.shuffle(cfg['shuffle_buffer_size']).batch(cfg['batch_size'][dataset_type])
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            time_3 = time.time()
            print(f'        Preparing TF datasets: took {(time_3-time_2):.1f} s.')

            # remove existing datasets
            path_to_dataset = to_absolute_path(f'datasets/{cfg.dataset_name}/{dataset_type}/{os.path.basename(file_name)}/{vs_type}')
            if os.path.exists(path_to_dataset):
                shutil.rmtree(path_to_dataset)
            else:
                os.makedirs(path_to_dataset, exist_ok=True)

            # save
            tf.data.experimental.save(dataset, path_to_dataset)
            OmegaConf.save(config=cfg, f=f'{path_to_dataset}/cfg.yaml')
            time_4 = time.time()
            print(f'        Saving TF datasets: took {(time_4-time_3):.1f} s.\n')

        print(f'\n-> Dataset ({dataset_type}) contains:')
        for k, v in n_samples.items():
            print(f'    {k}: {v} samples')

    print(f'Total time: {(time_4-time_start):.1f} s.\n') 

if __name__ == '__main__':
    main()