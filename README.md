# [ARCHIVED] Tatoo: a Tau Toolkit
**NB:** the repo is archived as the developments with additional updates have been merged into the core [TauAlgo group framework](https://github.com/cms-tau-pog/TauMLTools).
----- 

This repository is an area for R&D in ML-based tau lepton identification, in order to improve upon the existing baseline of the [DeepTau](https://arxiv.org/abs/2201.08458) architecture. The repository consists in standalone modules which assume to have as input files preprocessed within the core [TauAlgo group framework](https://github.com/cms-tau-pog/TauMLTools). The parts/modules are: 

* `create_dataset.py` -> preprocessing of the input ROOT files directly into TensorFlow ragged arrays with [`awkward`](https://awkward-array.org/doc/main/), optimised with `numba`.
* `train.py` -> on-the-fly composition of the training dataset via sampling across classes, optimised with unified dynamic batching. Experiment tracking with `mlflow`. 
* `models/`: adaptation of Transformer, [ParticleNet](https://arxiv.org/abs/1902.08570), and [Particle Convolutions](https://arxiv.org/abs/2107.02908), with custom embedding layer to handle heterogeneous input collections in a unified way. 
* `predict.py` -> model inference script.
* `plot_roc.py` -> plotting of per-class ROC curves with statistical uncertainty. 
* `visualization_notebook.ipynb` + `utils/visualize.py` -> visualisation of self-attention weights and corresponding particle interactions as a `plotly` widget.
