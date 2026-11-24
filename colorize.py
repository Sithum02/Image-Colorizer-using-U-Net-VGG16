"""Colorize grayscale (or colour) photos with the trained model.

  python colorize.py photo.jpg                      -> photo_colorized.jpg
  python colorize.py a.jpg b.png --out_dir outputs

The network works at 256x256. Its a, b prediction is resized to the size of the input and combined with the
input's own full-resolution L channel, so the output keeps the original detail.
"""
import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

from colorizer import IMG_SIZE, build_model, load_decoder, rgb_to_inputs


def colorize(model, path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)                  # gray files are read as three equal channels
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]

    small = tf.image.resize(rgb, (IMG_SIZE, IMG_SIZE), antialias=True)
    L_small, _ = rgb_to_inputs(tf.clip_by_value(small, 0.0, 1.0))
    ab = model.predict_on_batch(L_small[None])[0]
    ab = cv2.resize(ab, (w, h), interpolation=cv2.INTER_CUBIC) * 128.0

    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[..., :1]
    out = cv2.cvtColor(np.concatenate([L, ab], axis=-1).astype(np.float32), cv2.COLOR_Lab2RGB)
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('images', nargs='+')
    p.add_argument('--weights', default='weights/decoder_weights.npz')
    p.add_argument('--out_dir', default=None, help='default: next to each input')
    p.add_argument('--vgg_input', choices=['imagenet', 'unit'], default='unit')
    p.add_argument('--no_decoder_bn', dest='decoder_bn', action='store_false')
    args = p.parse_args()

    model = load_decoder(build_model(args.vgg_input, args.decoder_bn), args.weights)
    for path in args.images:
        stem, _ = os.path.splitext(os.path.basename(path))
        out_dir = args.out_dir or os.path.dirname(path) or '.'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{stem}_colorized.jpg')
        cv2.imwrite(out_path, cv2.cvtColor(colorize(model, path), cv2.COLOR_RGB2BGR))
        print(out_path)


if __name__ == '__main__':
    main()
