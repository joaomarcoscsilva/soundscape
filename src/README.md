# Source directory

## Modules
 - `constants.py`: Contains constants used in the project, like the classnames.
 - `utils.py`: Contains utility functions.
 - `dataset.py`: Contains constructors for various versions of the dataset.
 - `dataset_functions.py`: Contains tensorflow functions used for creating the dataset.
 - `data_fragmentation.py`: Contains functions for the division of minute-long clips into shorter fragments.
## Scripts
- `process_txts.py`: Generates a csv file with the labels for each audio file. Usage:

        python process_txts.py <path_to_txts_directory> <path_to_wavs_directory> <output-file>
 
- `check_labels.py`: Finds the percentage of labels in the `labels.csv` file that have no corresponding audio file in the directory passed. Usage:

        python check_labels.py <path_to_labels_file> <path_to_wavs_directory> 

- `precompute_spectrograms.py`: Precomputes the spectrograms for all audio files. Configuration in the dictionary inside the script. Usage:
        
        python precompute_spectrograms.py