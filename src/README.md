# Directory structure

 - `run_supervised.py`: Run a supervised experiment using the settings in `params.yaml`

 - `soundscape`
     - `data`
         - `dataset.py`: Constructors for various versions of the dataset.
         - `dataset_functions.py`: Tensorflow functions used for creating the dataset.
         - `data_fragmentation.py`: Functions for the division of minute-long clips into shorter fragments.
     
     - `lib`
         - `constants.py`: Constants used in the project, like the classnames.
         - `utils.py`: General utility functions
         - `settings.py`: Present the contents of the `params.yaml` file as a hashable dictionary.
         - `loss_transforms.py`: Functions for transforming the loss function, e.g. for class balancing or for generating a update function.
         - `model.py`: Functions for loading models and preparing them for training.
         - `sow_transforms.py`: Utilities for applying [`oryx`](https://github.com/jax-ml/oryx) transforms to functions, such as `sow` or `reap`. These are extensively used for passing around batchnorm statistics and metrics during training.
         - `train_loop.py`: Utilities for scanning through datasets during training and evaluation.
         - `log.py`: Utilities for logging metrics and model weights during training (closely tied to `train_loop.py`).
         - `supervised.py`: Main functions for running supervised experiments.
         
- `script`: Directory containing various scripts 

     - `process_txts.py`: Generate a csv file with the labels for each audio file. Usage:
             
            python process_txts.py <path_to_txts_directory> <path_to_wavs_directory> <output-file>

     - `precompute_spectrograms.py`: Precompute the spectrograms for all audio files. Configuration in the dictionary inside the script. Usage:

            python precompute_spectrograms.py                 

     - `split_labels.py`: Split a `labels.csv` file into train, validation and test sets. The proportions are set in `params.yaml`. Usage:

                python check_labels.py
 