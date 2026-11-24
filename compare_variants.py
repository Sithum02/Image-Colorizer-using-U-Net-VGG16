"""Short training runs that compare four versions of the model at the same budget:
encoder input in [0, 1] (as in the notebook) or in VGG16's ImageNet range, with or without batch
normalisation in the decoder.

  python compare_variants.py --data data/place365/DATA-1 --steps 1500 --out results/variant_comparison.json

Each variant trains for --steps batches from the same initial data order and is scored on the first 800
validation images. "saturated" is the share of output values with |a| or |b| > 0.99, i.e. where tanh is flat.
"""
import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

from colorizer import build_model, configure_gpu, list_images, make_dataset, split

VARIANTS = [('unit', False), ('unit', True), ('imagenet', False), ('imagenet', True)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/place365/DATA-1')
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--n_val', type=int, default=800)
    p.add_argument('--variants', nargs='*', default=None, help='e.g. unit:0 imagenet:1 (default: all four)')
    p.add_argument('--out', default='results/variant_comparison.json')
    p.add_argument('--gpu_memory_mb', type=int, default=None)
    args = p.parse_args()

    configure_gpu(args.gpu_memory_mb)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    parts = split(list_images(args.data))
    full = lambda rel: [os.path.join(args.data, r) for r in rel]
    val = list(make_dataset(full(parts['val'][:args.n_val]), 8))
    zero_mse = float(np.mean([np.mean(ab.numpy() ** 2) for _, ab in val]))

    variants = [(v.split(':')[0], v.split(':')[1] == '1') for v in args.variants] if args.variants else VARIANTS
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    results['zero_prediction_val_mse'] = zero_mse
    for vgg_input, bn in variants:
        tf.keras.backend.clear_session()
        tf.random.set_seed(0)
        train = iter(make_dataset(full(parts['train']), 8, shuffle=True, repeat=True, seed=0))
        model = build_model(vgg_input, bn)
        model.compile(tf.keras.optimizers.Adam(1e-4), 'mse')
        sat0 = float(np.mean(np.abs(model(val[0][0], training=False).numpy()) > 0.99))
        t = time.time()
        for _ in range(args.steps):
            model.train_on_batch(*next(train))
        val_mse = float(np.mean([model.test_on_batch(L, ab) for L, ab in val]))
        key = f'{vgg_input} input, decoder BN {"on" if bn else "off"}'
        results[key] = dict(steps=args.steps, val_mse=val_mse, vs_zero=val_mse / zero_mse - 1,
                            saturated_at_init=sat0,
                            saturated_after=float(np.mean(np.abs(model(val[0][0], training=False).numpy()) > 0.99)),
                            ms_per_step=1000 * (time.time() - t) / args.steps)
        print(key, json.dumps(results[key]), flush=True)
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(results, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
