"""Train the colorization model.

  python train.py --data data/place365/DATA-1 --out runs/main --steps_per_epoch 2000 --hours 4.25 --gpu_memory_mb 3000
  python train.py --data data/place365/DATA-1 --out runs/notebook --vgg_input unit --no_decoder_bn

Saves the best weights (lowest validation loss), the decoder weights as .npz, the per-epoch log and the split.
An interrupted run resumes from its last completed epoch when started again with the same --out.
"""
import argparse
import json
import os
import time

import tensorflow as tf

from colorizer import build_model, configure_gpu, list_images, make_dataset, save_decoder, split


class TimeBudget(tf.keras.callbacks.Callback):
    """Stop after the first epoch that ends past the time budget."""

    def __init__(self, hours):
        super().__init__()
        self.limit = hours * 3600 if hours else None
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if self.limit and time.time() - self.start > self.limit:
            print(f'\ntime budget reached after epoch {epoch + 1}')
            self.model.stop_training = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/place365/DATA-1')
    p.add_argument('--out', default='runs/main')
    p.add_argument('--vgg_input', choices=['imagenet', 'unit'], default='unit')
    p.add_argument('--no_decoder_bn', dest='decoder_bn', action='store_false')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--steps_per_epoch', type=int, default=None, help='default: one pass over the training set')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--hours', type=float, default=None)
    p.add_argument('--fp32', action='store_true', help='disable mixed precision')
    p.add_argument('--gpu_memory_mb', type=int, default=None)
    args = p.parse_args()

    configure_gpu(args.gpu_memory_mb)
    if not args.fp32:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    os.makedirs(args.out, exist_ok=True)

    parts = split(list_images(args.data))
    with open(os.path.join(args.out, 'split.json'), 'w') as f:
        json.dump(parts, f)
    full = lambda rel: [os.path.join(args.data, r) for r in rel]
    print({k: len(v) for k, v in parts.items()})

    steps = args.steps_per_epoch or len(parts['train']) // args.batch
    train_ds = make_dataset(full(parts['train']), args.batch, shuffle=True, repeat=True)
    val_ds = make_dataset(full(parts['val']), args.batch)

    model = build_model(args.vgg_input, args.decoder_bn)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss='mse', metrics=['mae'])
    print(f'trainable parameters: {sum(int(tf.size(w)) for w in model.trainable_weights):,}')

    callbacks = [
        tf.keras.callbacks.BackupAndRestore(os.path.join(args.out, 'backup')),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(args.out, 'best.h5'), monitor='val_loss',
                                           save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(os.path.join(args.out, 'train_log.csv'), append=True),
        TimeBudget(args.hours),
    ]
    start = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, steps_per_epoch=steps,
              callbacks=callbacks, verbose=2)

    model.load_weights(os.path.join(args.out, 'best.h5'))
    save_decoder(model, os.path.join(args.out, 'decoder_weights.npz'))
    with open(os.path.join(args.out, 'run.json'), 'w') as f:
        json.dump(dict(vars(args), steps_per_epoch=steps, train_hours=(time.time() - start) / 3600,
                       gpu=tf.config.list_physical_devices('GPU') != []), f, indent=1)
    print(f'done in {(time.time() - start) / 3600:.2f} h')


if __name__ == '__main__':
    main()
