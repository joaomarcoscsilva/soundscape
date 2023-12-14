@Composable
def tf2jax(batch):
    """
    Convert a dictionary of tensorflow tensors to a dictionary of jax arrays.
    """

    def _tf2jax(element):
        """
        Convert a single tensor to a jax array.
        """

        # Convert the tensor to a numpy array
        if isinstance(element, tf.Tensor):
            element = element.numpy()

        # if the dtype is adequate, convert to a jax array
        if isinstance(element, np.ndarray) and element.dtype != np.dtype("O"):
            element = jnp.array(element)

        return element

    batch = {k: _tf2jax(v) for k, v in batch.items()}

    return batch


@Composable
def split_rng(values):
    """
    Split the rng key into one key for each element in the batch.
    """

    rng = values["rng"]
    inputs = values["inputs"]

    rng, _rng = jax.random.split(rng, 2)
    rngs = jax.random.split(_rng, inputs.shape[0])

    return {**values, "rng": rng, "rngs": rngs}


# @Composable
# @settings_fn
# def split_image(values, *, split_shape):
#     """
#     Split a spectrogram into multiple smaller spectrograms.

#     Settings:
#     ---------
#     split_shape: int
#         The shape of the smaller spectrograms.
#     """

#     image = values["inputs"]

#     batch_size, height, width, channels = image.shape

#     new_image_shape = (batch_size, height, split_shape, channels)

#     image = image[:, :, :20 * split_shape]
#     image = tf.split(image, 20, axis=2)

#     return {**values, "inputs": image}


@Composable
@settings_fn
def split_image(values, *, split_shape):
    """
    Split a spectrogram into multiple smaller spectrograms.

    Settings:
    ---------
    split_shape: int
        The shape of the smaller spectrograms.
    """

    image = values["inputs"]

    height, width, channels = image.shape

    image = image[:, : 20 * split_shape]
    image = tf.split(image, 20, axis=1)

    if "id" in values:
        values["id"] = tf.repeat(values["id"], 20)

    if "labels" in values:
        values["labels"] = tf.repeat(values["labels"], 20)

    if "_file" in values:
        values["_file"] = tf.repeat(values["_file"], 20)

    return {**values, "inputs": image}


@Composable
@settings_fn
def prepare_image(values, *, precision):
    """
    Normalize a repeat an image's channels 3 times.

    Settings:
    ---------
    precision: int
        The number of bits used to represent each pixel.
    """

    image = values["inputs"]

    # Normalize the image
    image = jnp.float32(image) / (2**precision)

    # Repeat the channels 3 times
    image = jnp.repeat(image, 3, axis=-1)

    return {**values, "inputs": image}


@Composable
def downsample_image(values):
    """
    Downsample an image to 224x224 pixels.
    """

    image = values["inputs"]
    shape = (*image.shape[:-3], 224, 224, image.shape[-1])
    image = jax.image.resize(image, shape, method="bicubic")

    return {**values, "inputs": image}


@Composable
@settings_fn
def one_hot_encode(values, *, num_classes):
    """
    Convert a class name to a one-hot encoded vector

    Settings:
    ---------
    num_classes: int
        The number of classes to use. Can be 13, 12 or 2.
    """

    labels = values["labels"]

    labels = jax.nn.one_hot(labels, num_classes)

    return {**values, "one_hot_labels": labels}
