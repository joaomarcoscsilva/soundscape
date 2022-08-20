import tensorflow as tf
from functools import wraps
import os

import dataset_functions as dsfn
import data_fragmentation
import utils


def labels_dataset(settings):
    """
    Return a tf.data.Dataset containing the labelled audio files.
    """
    labels = dsfn.get_labels(settings)
    ds = tf.data.Dataset.from_tensor_slices(labels)
    return ds


def waveform_dataset(settings):
    """
    Return a tf.data.Dataset containing the labelled audio files and their waveforms.
    """
    ds = labels_dataset(settings)
    ds = ds.map(dsfn.get_extract_waveform_fn(settings))
    ds = ds.map(dsfn.get_fragment_borders_fn(settings))
    return ds


def melspectrogram_dataset(settings, from_disk=False):
    """
    Return a tf.data.Dataset containing the labelled audio files and their spectrograms.
    """
    if not from_disk:
        ds = waveform_dataset(settings)
        ds = ds.map(dsfn.get_extract_melspectrogram_fn(settings))
    else:
        ds = labels_dataset(settings)
        ds = ds.map(dsfn.get_fragment_borders_fn(settings))
        ds = ds.map(dsfn.get_load_melspectrogram_fn(settings))
    return ds


def get_jax_process_data_fn(settings):
    """
    Create a function that processes a batch of data
    from a tf.data.Dataset into a batch of data suitable
    for training with jax.
    """

    slice_fn = data_fragmentation.get_batch_slice_fn(settings)

    def jax_process_data(rng, args):

        args = utils.tf2jax(args)
        spec_frags = slice_fn(rng, args["spec"].T, args["frag_intervals"])

        return spec_frags

    return jax_process_data
