import os
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

from tensorflow.keras.callbacks import TensorBoard
from vit import VisionTransformer

AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--num-layers", default=8, type=int)
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--mlp-dim", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    args = parser.parse_args()

    ds = tfds.load("cifar10", as_supervised=True)
    ds_train = (
        ds["train"]
        .cache()
        .shuffle(1024)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )
    ds_test = (
        ds["test"]
        .cache()
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = VisionTransformer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_layers=args.num_layers,
            num_classes=10,
            d_model=args.d_model,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            channels=3,
            dropout=0.1,
        )
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            optimizer=tfa.optimizers.AdamW(
                learning_rate=args.lr, weight_decay=args.weight_decay
            ),
            metrics=["accuracy"],
        )

    early_stop = tf.keras.callbacks.EarlyStopping(patience=10),
    mcp = tf.keras.callbacks.ModelCheckpoint(filepath='weights/best.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0)    

    model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=args.epochs,
        callbacks=[early_stop, mcp, reduce_lr],
    )
    model.save_weights(os.path.join(args.logdir, "vit"))