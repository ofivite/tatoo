import awkward as ak
import tensorflow as tf

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

from utils.data_preprocessing import get_tau_arrays, preprocess_taus, awkward_to_ragged
from model.taco import TacoNet
from model.transformer import Transformer

# tf.config.set_visible_devices([], device_type='GPU')
tf.config.list_logical_devices()

import mlflow
from mlflow.tracking.context.git_context import _get_git_commit

@hydra.main(config_path='configs', config_name='train')
def main(cfg: DictConfig) -> None:

    # set up mlflow experiment id
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")
    experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
    if experiment is not None: # fetch existing experiment id
        run_kwargs = {'experiment_id': experiment.experiment_id}
    else: # create new experiment
        experiment_id = mlflow.create_experiment(cfg.experiment_name)
        run_kwargs = {'experiment_id': experiment_id}

    print('\n-> Retrieving input awkward arrays')
    a = get_tau_arrays(cfg.datasets, cfg.vs_type, cfg.tree_name)

    print('\n-> Preprocessing')
    a_train, a_val = preprocess_taus(a, cfg.vs_type, cfg.n_samples_train, cfg.n_samples_val)
    del(a)

    print('\n-> Preparing TF datasets')
    feature_name_to_idx = {name: cfg.feature_names.index(name) for name in cfg.feature_names}
    ragged_pf_train = awkward_to_ragged(a_train, cfg.feature_names)
    ragged_pf_val = awkward_to_ragged(a_val, cfg.feature_names)

    # add train labels
    train_labels = ak.to_pandas(a_train[['node_tau', f'node_{cfg.vs_type}']])
    print(train_labels.value_counts())
    train_labels = train_labels.values

    # create train data set
    train_data = tf.data.Dataset.from_tensor_slices((ragged_pf_train, train_labels))
    train_data = train_data.cache()
    train_data = train_data.shuffle(len(train_labels)).batch(cfg.batch_size)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

    # add validation labels
    val_labels = ak.to_pandas(a_val[['node_tau', f'node_{cfg.vs_type}']])
    print(val_labels.value_counts())
    val_labels = val_labels.values

    # create validation data set
    val_data = tf.data.Dataset.from_tensor_slices((ragged_pf_val, val_labels))
    val_data = val_data.cache()
    val_data = val_data.shuffle(len(val_labels)).batch(cfg.batch_size)
    val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)
    del(a_train, a_val)

    # start mlflow run
    with mlflow.start_run(**run_kwargs) as active_run:
        run_id = active_run.info.run_id
        
        # define model
        if cfg.model.type == 'taco_net':
            model = TacoNet(feature_name_to_idx, cfg.model.kwargs.encoder, cfg.model.kwargs.decoder)
        elif cfg.model.type == 'transformer':
            model = Transformer(**cfg.model.kwargs)
        else:
            raise RuntimeError('Failed to infer model type')
        
        model(ragged_pf_train[:1]) # build it for correct autologging with mlflow
        opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        model.compile(optimizer=opt,
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), 
                    metrics=['accuracy'])
        print(model.wave_encoder.summary())
        print(model.wave_decoder.summary())
        model.fit(train_data, validation_data=val_data, epochs=cfg.n_epochs, verbose=1)

         # save model
        print("\n-> Saving model")
        model.save((f'{cfg.model.name}.tf'), save_format="tf")
        mlflow.log_artifacts(f'{cfg.model.name}.tf', 'model')

        print("\n-> Evaluating performance")
        train_data_for_predict = tf.data.Dataset.from_tensor_slices(ragged_pf_train).batch(cfg.batch_size)
        val_data_for_predict = tf.data.Dataset.from_tensor_slices(ragged_pf_val).batch(cfg.batch_size)
        train_preds = model.predict(train_data_for_predict)
        val_preds = model.predict(val_data_for_predict)

        # compute metrics
        roc_auc_train = roc_auc_score(train_labels, train_preds)
        roc_auc_val = roc_auc_score(val_labels, val_preds)
        print('ROC AUC, train: ', roc_auc_train)
        print('ROC AUC, val: ', roc_auc_val)
        mlflow.log_metrics({'roc_auc': roc_auc_train, 'val_roc_auc': roc_auc_val})

        # log data params
        mlflow.log_param('vs_type', cfg.vs_type)
        mlflow.log_params({f'dataset_tau_{i}': dataset for i, dataset in enumerate(cfg.datasets['tau'].keys())})
        mlflow.log_params({f'dataset_vs_type_{i}': dataset for i, dataset in enumerate(cfg.datasets[cfg.vs_type].keys())})
        mlflow.log_params({'n_samples_train': cfg.n_samples_train, 'n_samples_val': cfg.n_samples_val})

        # log model params
        mlflow.log_param('model_name', cfg.model.name)
        if cfg.model.type == 'taco_net':
            mlflow.log_params(cfg.model.kwargs.encoder)
            mlflow.log_params(cfg.model.kwargs.decoder)
        elif cfg.model.type == 'transformer':
            mlflow.log_params(cfg.model.kwargs)
        
        # log N trainable params 
        summary_list = []
        model.summary(print_fn=summary_list.append)
        for l in summary_list:
            if (s:='Trainable params: ') in l:
                mlflow.log_param('n_train_params', int(l.split(s)[-1].replace(',', '')))
        
        # log encoder & decoder summaries
        if cfg.model.type == 'taco_net':
            summary_list_encoder, summary_list_decoder = [], []
            model.wave_encoder.summary(print_fn=summary_list_encoder.append)
            model.wave_decoder.summary(print_fn=summary_list_decoder.append)
            summary_encoder, summary_decoder = "\n".join(summary_list_encoder), "\n".join(summary_list_decoder)
            mlflow.log_text(summary_encoder, artifact_file="encoder_summary.txt")
            mlflow.log_text(summary_decoder, artifact_file="decoder_summary.txt") 

        # log misc. info
        mlflow.log_param('run_id', run_id)
        mlflow.log_param('git_commit', _get_git_commit(to_absolute_path('.')))

if __name__ == '__main__':
    main()