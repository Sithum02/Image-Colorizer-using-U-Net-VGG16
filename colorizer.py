"""Data pipeline and model for predicting the colour (a, b) of a grayscale image (L) in CIELAB space.

The architecture is the one in Automatic_Colourizer.ipynb: a frozen ImageNet VGG16 encoder, skip connections
from four of its blocks and a four-stage upsampling decoder. Preprocessing runs as TensorFlow ops so that the
input pipeline keeps up with the GPU.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

IMG_SIZE = 256
VGG_MEAN_BGR = (103.939, 116.779, 123.68)

# sRGB (D65) -> XYZ, the same matrix and white point OpenCV uses for COLOR_RGB2Lab
_RGB2XYZ = np.array([[0.412453, 0.357580, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]], dtype=np.float32)
_WHITE = np.array([0.950456, 1.0, 1.088754], dtype=np.float32)


def configure_gpu(memory_mb=None):
    """Optionally give TensorFlow an explicit memory budget (on Windows it otherwise takes about half the free
    memory of a small GPU)."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and memory_mb:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=memory_mb)])


# ---------------------------------------------------------------------------------------------- data

def rgb_to_lab(rgb):
    """sRGB in [0, 1], shape (..., 3) -> CIELAB with L in [0, 100] and a, b in about [-128, 127]."""
    lin = tf.where(rgb > 0.04045, tf.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
    xyz = tf.tensordot(lin, tf.constant(_RGB2XYZ.T), axes=1) / tf.constant(_WHITE)
    f = tf.where(xyz > 0.008856, tf.pow(tf.maximum(xyz, 1e-12), 1.0 / 3.0), 7.787 * xyz + 16.0 / 116.0)
    y = xyz[..., 1]
    L = tf.where(y > 0.008856, 116.0 * f[..., 1] - 16.0, 903.3 * y)
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return tf.stack([L, a, b], axis=-1)


def load_rgb(path):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), antialias=True) / 255.0
    return tf.clip_by_value(img, 0.0, 1.0)


def rgb_to_inputs(rgb):
    """Network input and target, normalised as in the notebook: L / 100 in [0, 1], a / 128 and b / 128 in [-1, 1]."""
    lab = rgb_to_lab(rgb)
    return lab[..., :1] / 100.0, lab[..., 1:] / 128.0


def load_example(path):
    return rgb_to_inputs(load_rgb(path))


def make_dataset(paths, batch_size, shuffle=False, repeat=False, seed=0):
    ds = tf.data.Dataset.from_tensor_slices(list(paths))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 5000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load_example, num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    return ds.batch(batch_size, drop_remainder=shuffle).prefetch(tf.data.AUTOTUNE)


def list_images(root):
    """All .jpg files under root, as paths relative to root, sorted so the split is the same on every machine."""
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('*.jpg'))


def split(rel_paths, n_test=2500, n_val=2500, seed=0):
    idx = np.random.default_rng(seed).permutation(len(rel_paths))
    pick = lambda ids: sorted(rel_paths[i] for i in ids)
    return dict(test=pick(idx[:n_test]), val=pick(idx[n_test:n_test + n_val]), train=pick(idx[n_test + n_val:]))


# --------------------------------------------------------------------------------------------- model

def build_model(vgg_input='unit', decoder_bn=True):
    """vgg_input='imagenet' feeds VGG16 the range it was trained on (0-255, ImageNet mean removed);
    'unit' feeds L in [0, 1] repeated to three channels, as the original notebook did.
    decoder_bn adds batch normalisation after each decoder convolution (the notebook had none)."""
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1), name='L_input')
    if vgg_input == 'imagenet':
        x = layers.Lambda(lambda t: tf.concat([t, t, t], -1) * 255.0 - tf.constant(VGG_MEAN_BGR, dtype=t.dtype),
                          name='vgg_preprocess')(inputs)
    elif vgg_input == 'unit':
        x = layers.Concatenate(name='gray_to_rgb')([inputs, inputs, inputs])
    else:
        raise ValueError(vgg_input)

    vgg = VGG16(include_top=False, weights='imagenet', input_tensor=x)
    for layer in vgg.layers:
        layer.trainable = False
    skips = [vgg.get_layer(n).output for n in ('block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2')]
    x = vgg.get_layer('block5_conv3').output                                  # 16x16 bottleneck

    for i, (skip, ch) in enumerate(zip(skips, (512, 256, 128, 64)), start=1):  # 16 -> 32 -> 64 -> 128 -> 256
        x = layers.UpSampling2D(2, name=f'dec{i}_up')(x)
        x = layers.Concatenate(name=f'dec{i}_concat')([x, skip])
        for j in (1, 2):
            if decoder_bn:
                x = layers.Conv2D(ch, 3, padding='same', use_bias=False, name=f'dec{i}_conv{j}')(x)
                x = layers.BatchNormalization(name=f'dec{i}_bn{j}')(x)
                x = layers.ReLU(name=f'dec{i}_relu{j}')(x)
            else:
                x = layers.Conv2D(ch, 3, activation='relu', padding='same', name=f'dec{i}_conv{j}')(x)
    outputs = layers.Conv2D(2, 1, activation='tanh', padding='same', dtype='float32', name='ab_output')(x)
    return models.Model(inputs, outputs, name='Colorization_UNet_VGG16')


def _key(var):
    return var.name.split(':')[0].replace('/', '.')          # 'dec1_conv1/kernel:0' -> 'dec1_conv1.kernel'


def _decoder_weights(model):
    return [w for w in model.weights if not w.name.startswith('block')]   # VGG16 layers are named block*


def save_decoder(model, path):
    """Only the decoder is stored (including batch-norm statistics); the frozen encoder comes from the Keras
    ImageNet weights."""
    np.savez_compressed(path, **{_key(w): w.numpy() for w in _decoder_weights(model)})


def load_decoder(model, path):
    data = np.load(path)
    for w in _decoder_weights(model):
        w.assign(data[_key(w)])
    return model
