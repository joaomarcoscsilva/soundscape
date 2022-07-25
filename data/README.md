# Data directory

## Files
 - `data/constants.py`: Contains constants used in the project, like the classnames.
 - `data/process_txts.py`: Generates a `labels.csv` file with the labels for each audio file. Takes in the path to a directory with the labels in `.txt` files.
 - `data/check_labels.py`: Finds the percentage of labels in the `labels.csv` file that have no corresponding audio file in the directory passed.