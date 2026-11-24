"""Evaluate a trained model on the held-out test images and draw the figures.

  python evaluate.py --data data/place365/DATA-1 --run runs/main --out results

Metrics on the reconstructed RGB images (256x256), against the original photos:
  PSNR and SSIM, for the model and for a grayscale baseline (the same L with a = b = 0);
  error on the colour channels a, b (MSE and MAE in the normalised [-1, 1] units used for training);
  colourfulness (Hasler and Suesstrunk, 2003) of the predictions and of the originals.
Means over the test set, per group (artificial / natural) and per scene category, are written to metrics.json and
the per-image scores to test_per_image.csv. Some test photos are black and white to begin with (colourfulness of the
original below 5); the gray baseline reproduces those almost exactly, so the test set is also summarised without them.
"""
import argparse
import csv
import json
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from colorizer import build_model, configure_gpu, load_decoder, load_rgb, rgb_to_inputs

GRAY_PHOTO = 5.0   # originals less colourful than this are treated as black-and-white photos


def lab_to_rgb(L, ab):
    """Network units (L in [0, 1], ab in [-1, 1]) -> RGB in [0, 1]."""
    lab = np.concatenate([L * 100.0, ab * 128.0], axis=-1).astype(np.float32)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0.0, 1.0)


def colourfulness(rgb):
    r, g, b = (rgb * 255.0).transpose(2, 0, 1)
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def scores(pred, ref):
    return (peak_signal_noise_ratio(ref, pred, data_range=1.0),
            structural_similarity(ref, pred, channel_axis=2, data_range=1.0))


def summarise(rows, keys):
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out['psnr_median'] = float(np.median([r['psnr'] for r in rows]))
    out['psnr_gray_median'] = float(np.median([r['psnr_gray'] for r in rows]))
    out['model_beats_gray_psnr'] = float(np.mean([r['psnr'] > r['psnr_gray'] for r in rows]))
    return out | {'n': len(rows)}


def plot_curves(log_path, out_path, steps, batch):
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    ep = [int(r['epoch']) + 1 for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(ep, [float(r['loss']) for r in rows], 'o-', label='training')
    ax.plot(ep, [float(r['val_loss']) for r in rows], 's-', label='validation')
    ax.set_xlabel(f'epoch ({steps} steps of {batch} images)')
    ax.set_ylabel('MSE on a, b (normalised)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_examples(model, data, rel_paths, out_path, labels=None):
    """One row per image: the L input, the prediction and the original."""
    fig, axes = plt.subplots(len(rel_paths), 3, figsize=(7.5, 2.5 * len(rel_paths)), squeeze=False)
    for n, (row, rel) in enumerate(zip(axes, rel_paths)):
        rgb = load_rgb(os.path.join(data, rel))
        L, _ = rgb_to_inputs(rgb)
        pab = model.predict_on_batch(L[None])[0]
        L = L.numpy()
        imgs = [L[..., 0], lab_to_rgb(L, pab), rgb.numpy()]
        for ax, img, title in zip(row, imgs, ['input (L)', 'predicted', 'original']):
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if n == 0:
                ax.set_title(title)
        if labels:
            row[0].set_ylabel(labels[n], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/place365/DATA-1')
    p.add_argument('--run', default='runs/main')
    p.add_argument('--out', default='results')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--examples', type=int, default=8)
    p.add_argument('--gpu_memory_mb', type=int, default=None)
    args = p.parse_args()
    configure_gpu(args.gpu_memory_mb)
    os.makedirs(args.out, exist_ok=True)

    run = json.load(open(os.path.join(args.run, 'run.json')))
    test = json.load(open(os.path.join(args.run, 'split.json')))['test']
    model = load_decoder(build_model(run['vgg_input'], run['decoder_bn']), os.path.join(args.run, 'decoder_weights.npz'))

    paths = tf.data.Dataset.from_tensor_slices([os.path.join(args.data, r) for r in test])
    ds = paths.map(load_rgb, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch).prefetch(tf.data.AUTOTUNE)

    rows, i = [], 0
    for rgb in ds:
        L, ab = rgb_to_inputs(rgb)
        pred_ab = model.predict_on_batch(L)
        for rgb_i, L_i, ab_i, pab_i in zip(rgb.numpy(), L.numpy(), ab.numpy(), pred_ab):
            pred = lab_to_rgb(L_i, pab_i)
            gray = lab_to_rgb(L_i, np.zeros_like(pab_i))
            psnr, ssim = scores(pred, rgb_i)
            psnr_g, ssim_g = scores(gray, rgb_i)
            rows.append(dict(path=test[i], group=test[i].split('/')[0], category=test[i].split('/')[1],
                             psnr=psnr, ssim=ssim, psnr_gray=psnr_g, ssim_gray=ssim_g,
                             ab_mse=float(np.mean((pab_i - ab_i) ** 2)), ab_mse_zero=float(np.mean(ab_i ** 2)),
                             ab_mae=float(np.mean(np.abs(pab_i - ab_i))), ab_mae_zero=float(np.mean(np.abs(ab_i))),
                             colourfulness_pred=colourfulness(pred), colourfulness_true=colourfulness(rgb_i)))
            i += 1

    keys = ['psnr', 'psnr_gray', 'ssim', 'ssim_gray', 'ab_mse', 'ab_mse_zero', 'ab_mae', 'ab_mae_zero',
            'colourfulness_pred', 'colourfulness_true']
    by = lambda field: {g: summarise([r for r in rows if r[field] == g], keys) for g in sorted({r[field] for r in rows})}
    colour = [r for r in rows if r['colourfulness_true'] >= GRAY_PHOTO]
    out = dict(run=run, test=summarise(rows, keys), test_colour_photos=summarise(colour, keys),
               by_group=by('group'), by_category=by('category'))
    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(args.out, 'test_per_image.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    print(json.dumps(out['test'], indent=1))

    # examples: a fixed random draw from the test set (not hand-picked)
    pick = sorted(np.random.default_rng(1).choice(len(test), args.examples, replace=False))
    plot_examples(model, args.data, [test[k] for k in pick], os.path.join(args.out, 'examples.jpg'))

    # the colour test photos where the prediction loses most PSNR against leaving the image gray
    worst = sorted(colour, key=lambda r: r['psnr'] - r['psnr_gray'])[:4]
    plot_examples(model, args.data, [r['path'] for r in worst], os.path.join(args.out, 'failures.jpg'),
                  labels=[f"{r['category'].replace('_', ' ')}\n{r['psnr']:.1f} dB (gray {r['psnr_gray']:.1f} dB)" for r in worst])

    plot_curves(os.path.join(args.run, 'train_log.csv'), os.path.join(args.out, 'training_curves.png'),
                run['steps_per_epoch'], run['batch'])
    print(f"saved metrics.json, test_per_image.csv, examples.jpg, failures.jpg, training_curves.png in {args.out}")


if __name__ == '__main__':
    main()
