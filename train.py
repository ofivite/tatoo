import shutil
from glob import glob
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import roc_auc_score

import tensorflow as tf

from models.taco import TacoNet
from models.transformer import Transformer

import mlflow
from mlflow.tracking.context.git_context import _get_git_commit
mlflow.tensorflow.autolog(log_models=False) 

@hydra.main(config_path='configs', config_name='train')
def main(cfg: DictConfig) -> None:

    # setup gpu
    physical_devices = tf.config.list_physical_devices('GPU') 
    # tf.config.experimental.set_memory_growth(physical_devices[cfg["gpu_id"]], True)
    tf.config.set_logical_device_configuration(
            physical_devices[cfg["gpu_id"]],
            [tf.config.LogicalDeviceConfiguration(memory_limit=cfg["memory_limit"]*1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    # set up mlflow experiment id
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")
    experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
    if experiment is not None: # fetch existing experiment id
        run_kwargs = {'experiment_id': experiment.experiment_id}
    else: # create new experiment
        experiment_id = mlflow.create_experiment(cfg.experiment_name)
        run_kwargs = {'experiment_id': experiment_id}

    # start mlflow run
    with mlflow.start_run(**run_kwargs) as active_run:
        run_id = active_run.info.run_id
        
        # load datasets 
        train_datasets, val_datasets = [], []
        for p in glob(to_absolute_path(f'datasets/{cfg.dataset_name}/train/ShuffleMergeSpectral_*/{cfg.vs_type}')):
            dataset = tf.data.experimental.load(p)#.repeat()
            train_datasets.append(dataset) 
        for p in glob(to_absolute_path(f'datasets/{cfg.dataset_name}/val/ShuffleMergeSpectral_*/{cfg.vs_type}')):
            dataset = tf.data.experimental.load(p)
            val_datasets.append(dataset) 
        train_data = tf.data.Dataset.sample_from_datasets(datasets=train_datasets, stop_on_empty_dataset=False)
        val_data = tf.data.Dataset.sample_from_datasets(datasets=val_datasets, stop_on_empty_dataset=False)
        # train_data = tf.data.Dataset.choose_from_datasets(train_datasets, choice_dataset=tf.data.Dataset.range(len(train_datasets)), stop_on_empty_dataset=False)
        # val_data = tf.data.Dataset.choose_from_datasets(val_datasets, choice_dataset=tf.data.Dataset.range(len(val_datasets)), stop_on_empty_dataset=False)

        # train_data = tf.data.experimental.load(to_absolute_path(f'datasets/{cfg.dataset_name}/train/ShuffleMergeSpectral_{cfg.dataset_id}/{cfg.vs_type}'))
        # val_data = tf.data.experimental.load(to_absolute_path(f'datasets/{cfg.dataset_name}/val/ShuffleMergeSpectral_219/{cfg.vs_type}'))

        # define model
        feature_name_to_idx = {name: cfg.feature_names.index(name) for name in cfg.feature_names}
        if cfg.model.type == 'taco_net':
            model = TacoNet(feature_name_to_idx, cfg.model.kwargs.encoder, cfg.model.kwargs.decoder)
        elif cfg.model.type == 'transformer':
            model = Transformer(feature_name_to_idx, cfg.model.kwargs.encoder, cfg.model.kwargs.decoder)
        else:
            raise RuntimeError('Failed to infer model type')
        X_, _ = next(iter(train_data))
        model(X_) # init it for correct autologging with mlflow

        # compile and fit
        opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, mode='auto', restore_best_weights=True)
        checkpoint_path = 'tmp_checkpoints'
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + "/" + "epoch_{epoch:02d}---val_loss_{val_loss:.3f}",
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_freq='epoch',
            save_best_only=False)
        callbacks = [early_stopping, model_checkpoint]
        model.compile(optimizer=opt,
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), 
                    metrics=['accuracy', tf.keras.metrics.AUC(from_logits=False)])
        model.fit(train_data, validation_data=val_data, epochs=cfg.n_epochs, callbacks=callbacks, verbose=1)  #  steps_per_epoch=1000, 

        # save model
        print("\n-> Saving model")
        model.save((f'{cfg.model.name}.tf'), save_format="tf")
        mlflow.log_artifacts(f'{cfg.model.name}.tf', 'model')
        if cfg.model.type == 'taco_net':
            print(model.wave_encoder.summary())
            print(model.wave_decoder.summary())
        elif cfg.model.type == 'transformer':
            print(model.summary())

        # log data params
        mlflow.log_param('vs_type', cfg.vs_type)
        mlflow.log_param('dataset_name', cfg.dataset_name)

        # log model params
        params_encoder = OmegaConf.to_object(cfg.model.kwargs.encoder)
        params_embedding = params_encoder.pop('embedding_kwargs')
        params_embedding = {f'emb_{p}': v for p,v in params_embedding.items()}
        mlflow.log_param('model_name', cfg.model.name)
        mlflow.log_params(params_encoder)
        mlflow.log_params(params_embedding)
        mlflow.log_params(cfg.model.kwargs.decoder)
        
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
        mlflow.log_artifacts(checkpoint_path, "checkpoints")
        shutil.rmtree(checkpoint_path)

        print(f'\nTraining has finished! Corresponding MLflow experiment name (ID): {cfg.experiment_name}({run_kwargs["experiment_id"]}), and run ID: {run_id}\n')

if __name__ == '__main__':
    main()