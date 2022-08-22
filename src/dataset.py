import tensorflow as tf
import jax
import numpy as np

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


def add_rng(ds, rng):
    """
    Add a random seed zipped with each element in the dataset.
    """

    rngs = jax.random.split(rng, len(ds))
    rngs = np.array(rngs)
    rngs = tf.constant(rngs)

    def add_rng(args):
        return {"rng": rngs[args["index"]], **args}

    return ds.map(add_rng)


def fragment_dataset(settings, ds, rng):
    """
    Fragments a dataset into fragments of the size specified in settings.
    This function must be called every epoch with different rng values.
    Note that train/validation/test splits must be done before calling this function.
    """

    ds = add_rng(ds, rng)

    (
        fn,
        flatten_inputs_fn,
        unflatten_outputs_fn,
        output_types,
    ) = data_fragmentation.get_jax_fragmentation_fn(settings)

    def f(args):
        flattened_args = flatten_inputs_fn(args)
        flattened_args = tf.py_function(fn, flattened_args, output_types)
        args = unflatten_outputs_fn(flattened_args)
        return tf.data.Dataset.from_tensor_slices(args)

    ds = ds.flat_map(f)

    num_samples = dsfn.get_labels(settings)["num_events"].sum()
    ds = ds.apply(tf.data.experimental.assert_cardinality(num_samples))

    return ds
