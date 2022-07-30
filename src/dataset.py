import tensorflow as tf

import dataset_functions as dsfn


def labels_dataset(settings):
    """
    Returns a tf.data.Dataset containing the labelled audio files.
    """
    labels = dsfn.get_labels(settings)
    ds = tf.data.Dataset.from_tensor_slices(labels)
    return ds


def waveform_dataset(settings):
    """
    Returns a tf.data.Dataset containing the labelled audio files and their waveforms.
    """
    ds = labels_dataset(settings)
    ds = ds.map(dsfn.extract_waveform_fn(settings))
    ds = ds.map(dsfn.fragment_borders_fn(settings))
    return ds


def melspectrogram_dataset(settings):
    """
    Returns a tf.data.Dataset containing the labelled audio files and their spectrograms.
    """
    ds = waveform_dataset(settings)
    ds = ds.map(dsfn.extract_melspectrogram_fn(settings))
    return ds
