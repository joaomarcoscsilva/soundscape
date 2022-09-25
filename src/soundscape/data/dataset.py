import tensorflow as tf
import jax
import numpy as np

from . import dataset_functions as dsfn
from . import data_fragmentation
from ..lib import utils


def labels_dataset(path="labels.csv"):
    """
    Return a tf.data.Dataset containing the labelled audio files.
    """

    labels = dsfn.get_labels(path)
    ds = tf.data.Dataset.from_tensor_slices(labels)

    return ds


def waveform_dataset(path="labels.csv"):
    """
    Return a tf.data.Dataset containing the labelled audio files and their waveforms.
    """

    ds = labels_dataset(path)
    ds = ds.map(dsfn.extract_waveform)
    ds = ds.map(dsfn.fragment_borders)
    return ds


def spectrogram_dataset(from_disk=False, path="labels.csv"):
    """
    Return a tf.data.Dataset containing the labelled audio files and their spectrograms.
    """

    if not from_disk:
        ds = waveform_dataset(path)
        ds = ds.map(dsfn.extract_spectrogram)
    else:
        ds = labels_dataset(path)
        ds = ds.map(dsfn.fragment_borders)
        ds = ds.map(dsfn.load_spectrogram)

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


def fragment_dataset(ds, rng, frag_key="spec"):
    """
    Fragments a dataset into fragments of the size specified in settings.
    This function must be called every epoch with different rng values.
    Note that train/validation/test splits must be done before calling this function.
    """

    ds = add_rng(ds, rng)

    input_keys = {
        "filename",
        "labels",
        "num_events",
        "index",
        "frag_intervals",
        "freq_intervals",
        "time_intervals",
        "rng",
        frag_key,
    }

    output_types = {
        "filename": tf.string,
        "labels": tf.int32,
        "num_events": tf.int32,
        "index": tf.int32,
        "frag_intervals": tf.float32,
        "freq_intervals": tf.float32,
        "time_intervals": tf.float32,
        frag_key: tf.uint16 if frag_key == "spec" else tf.float32,
    }

    fn = utils.tf_py_func(
        data_fragmentation.dict_slice_fragments, input_keys, output_types
    )

    # TODO: see if performance improves by replacing these 2 lines with a single flat_map
    ds = ds.map(fn)
    ds = ds.unbatch()

    return ds


def jax_dataset(ds, resize=None):
    """
    Convert a tf.data.Dataset to an iterator that returns jax arrays.
    Uses utils.tf2jax to do the conversions
    """

    for elem in ds:
        jax_args = utils.tf2jax(elem, to_gpu=True)
        yield jax_args
