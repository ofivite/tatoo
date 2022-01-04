import awkward as ak
import numpy as np
import tensorflow as tf

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

from utils.data_preprocessing import get_tau_arrays, preprocess_taus, awkward_to_ragged
from model.taco import TacoNet

tf.config.set_visible_devices([], device_type='GPU')
tf.config.list_logical_devices()


@hydra.main(config_path='.', config_name='train')
def main(cfg: DictConfig) -> None:

    print('\n-> Retrieving input awkward arrays')
    a = get_tau_arrays(cfg.datasets, cfg.tree_name)
    
    print('\n-> Preprocessing')
    a = preprocess_taus(a)
    a_taus = a[a['node_tau'] == 1]
    a_jets = a[a['node_jet'] == 1]
    a_train = ak.concatenate([a_taus[:cfg.n_samples_train], a_jets[:cfg.n_samples_train]], axis=0)
    a_val = ak.concatenate([a_taus[cfg.n_samples_train:cfg.n_samples_train+cfg.n_samples_val], \
                            a_jets[cfg.n_samples_train:cfg.n_samples_train+cfg.n_samples_val]], axis=0)
    a_train = a_train[np.random.permutation(len(a_train))]
    a_val = a_val[np.random.permutation(len(a_val))]
    
    print('\n-> Preparing TF datasets')
    feature_name_to_idx = {name: cfg.feature_names.index(name) for name in cfg.feature_names}
    ragged_pf_train = awkward_to_ragged(a_train, cfg.feature_names)
    ragged_pf_val = awkward_to_ragged(a_val, cfg.feature_names)
    del(a_taus, a_jets, a)

    # add train labels
    train_labels = ak.to_pandas(a_train[['node_tau', 'node_jet']])
    print(train_labels.value_counts())
    train_labels = train_labels.values

    # create train data set
    train_data = tf.data.Dataset.from_tensor_slices((ragged_pf_train, train_labels))
    train_data = train_data.cache()
    train_data = train_data.shuffle(len(train_labels)).batch(cfg.batch_size)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

    # add validation labels
    val_labels = ak.to_pandas(a_val[['node_tau', 'node_jet']])
    print(val_labels.value_counts())
    val_labels = val_labels.values

    # create validation data set
    val_data = tf.data.Dataset.from_tensor_slices((ragged_pf_val, val_labels))
    val_data = val_data.cache()
    val_data = val_data.shuffle(len(val_labels)).batch(cfg.batch_size)
    val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)
    del(a_train, a_val)

    # define model
    model = TacoNet(feature_name_to_idx, cfg.encoder_args.hidden_dim_encoder, cfg.encoder_args.n_freqs, cfg.encoder_args.n_filters, cfg.encoder_args.n_rotations, 
                        cfg.decoder_args.kernel_size, cfg.decoder_args.n_conv_filters, cfg.decoder_args.hidden_dim_decoder, cfg.decoder_args.n_outputs)

    opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(optimizer=opt,
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), 
                metrics=['accuracy'])

    model.fit(train_data, validation_data=val_data, epochs=cfg.n_epochs, verbose=1)

    # compute metrics
    print("\n-> Evaluating performance")
    train_data_for_predict = tf.data.Dataset.from_tensor_slices(ragged_pf_train).batch(10000)
    val_data_for_predict = tf.data.Dataset.from_tensor_slices(ragged_pf_val).batch(10000)
    train_preds = model.predict(train_data_for_predict)
    val_preds = model.predict(val_data_for_predict)
    print('ROC AUC, train: ', roc_auc_score(train_labels, train_preds))
    print('ROC AUC, val: ', roc_auc_score(val_labels, val_preds))

    # save model
    print("\n-> Saving model")
    model.save(to_absolute_path(f'models/{cfg.model_name}.tf'), save_format="tf")

if __name__ == '__main__':
    main()