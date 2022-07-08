import yaml
from glob import glob
from collections import defaultdict
from hydra.utils import to_absolute_path

import tensorflow as tf

def compose_datasets(datasets, tf_dataset_cfg):
    train_probas = [] # to store sampling probabilites on training datasets
    datasets_for_training = {'train': [], 'val': []} # to accumulate final datasets
    for dataset_type in datasets_for_training.keys():
        ds_per_tau_type = defaultdict(list)
        if dataset_type not in datasets:
            raise RuntimeError(f'key ({dataset_type}) should be present in dataset yaml configuration')
        for dataset_name, dataset_cfg in datasets[dataset_type].items(): # loop over specified train/val datasets
            for tau_type in dataset_cfg["tau_types"]: # loop over tau types specified for this dataset
                for p in glob(to_absolute_path(f'{dataset_cfg["path_to_dataset"]}/{dataset_name}/{dataset_type}/*/{tau_type}')): # loop over all globbed files in the dataset
                    dataset = tf.data.experimental.load(p) 
                    ds_per_tau_type[tau_type].append(dataset) # add TF dataset (1 input file, 1 tau type) to the map  
        
        n_tau_types = len(ds_per_tau_type.keys())
        for tau_type, ds_list in ds_per_tau_type.items():
            datasets_for_training[dataset_type] += ds_list # add datasets to the final list
            if dataset_type == "train": # for training dataset also keep corresponding sampling probas over input files
                n_files = len(ds_list)
                train_probas += n_files*[1./(n_tau_types*n_files) ]

    assert round(sum(train_probas), 5) == 1
    train_data = tf.data.Dataset.sample_from_datasets(datasets=datasets_for_training['train'], weights=train_probas, seed=1234, stop_on_empty_dataset=False) # True so that the last batches are not purely of one class
    val_data = tf.data.Dataset.sample_from_datasets(datasets=datasets_for_training['val'], seed=1234, stop_on_empty_dataset=False)

    # form TF datasets
    if tf_dataset_cfg["shuffle_buffer_size"] is not None:
        train_data = train_data.shuffle(tf_dataset_cfg["shuffle_buffer_size"])
    if tf_dataset_cfg["cache"]:
        train_data = train_data.cache()
    train_data = train_data.batch(tf_dataset_cfg["train_batch_size"])
    val_data = val_data.batch(tf_dataset_cfg["val_batch_size"])
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)

    # select from stored labels only those classes which are specified in the cfg 
    class_idx = {}
    for dataset_type in ['train', 'val']:
        dataset_name = list(datasets[dataset_type].keys())[0] # assume that label structure is the same in all datasets, so retrieve from the first one
        path_to_dataset = datasets[dataset_type][dataset_name]["path_to_dataset"]
        p = glob(to_absolute_path(f'{path_to_dataset}/{dataset_name}/{dataset_type}/*/tau'))[0] # load one dataset cfg
        with open(f'{p}/cfg.yaml', 'r') as f:
            data_cfg = yaml.safe_load(f)
        class_idx[dataset_type] = [data_cfg["label_columns"].index(f'label_{c}') for c in tf_dataset_cfg["classes"]] # fetch label indices which correspond to specified classes
    
    train_data = train_data.map(lambda X,y: (X, tf.gather(y, indices=class_idx['train'], axis=-1)),
                                num_parallel_calls=tf.data.AUTOTUNE) 
    val_data = val_data.map(lambda X,y: (X, tf.gather(y, indices=class_idx['val'], axis=-1)),  
                                num_parallel_calls=tf.data.AUTOTUNE) 

    return train_data, val_data