# Source directory

## Modules
 - `constants.py`: Contains constants used in the project, like the classnames.
 - `dataset.py`: Contains the code for the data loaders used.

## Scripts
- `process_txts.py`: Generates a csv file with the labels for each audio file. Usage:

        python process_txts.py <path_to_txts_directory> <path_to_wavs_directory> <output-file>
 
- `check_labels.py`: Finds the percentage of labels in the `labels.csv` file that have no corresponding audio file in the directory passed. Usage:

        python check_labels.py <path_to_labels_file> <path_to_wavs_directory> 