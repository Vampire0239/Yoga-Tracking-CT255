# -*- coding: utf-8 -*-
# CNN ảnh RGB chống overfitting (yoga pose)
# Biện pháp: GAP thay Flatten, L2 regularization, Dropout & SpatialDropout2D,
# Label Smoothing, MixUp tùy chọn, EarlyStopping/Checkpoint theo val_loss

import os, json, argparse, math
import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks, optimizers, regularizers

# ========== Cấu hình mặc định ==========
IMG_SIZE    = (256, 256)
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-4
WD          = 1e-4          # weight decay (L2)
DROP        = 0.5
LABEL_SMOOTH= 0.05
MIXUP_ALPHA = 0.2           # 0 = tắt; >0 = bật MixUp

# ========== Dataloader ==========
def build_datasets(data_root, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_dir = os.path.join(data_root, "train")
    # ưu tiên "val", nếu không có thì dùng "test"
    val_dir = None
    for d in ["val", "test"]:
        p = os.path.join(data_root, d)
        if os.path.isdir(p):
            val_dir = p; break
    if val_dir is None:
        raise FileNotFoundError("Không thấy data/val hoặc data/test")

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.0625,
        height_shift_range=0.0625,
        zoom_range=[0.5, 1.5],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode="categorical", shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size,
        class_mode="categorical", shuffle=False
    )
    return train_gen, val_gen

# ========== MixUp cho ImageDataGenerator ==========
def mixup_generator(generator, alpha=0.2):
    if alpha <= 0:
        while True:
            yield next(generator)
    else:
        while True:
            x1, y1 = next(generator)
            x2, y2 = next(generator)

            # Cắt về cùng size để tránh broadcast lỗi
            n = min(len(x1), len(x2))
            x1, y1 = x1[:n], y1[:n]
            x2, y2 = x2[:n], y2[:n]

            lam = np.random.beta(alpha, alpha, size=(n,)).astype(np.float32)
            lam_x = lam.reshape((-1, 1, 1, 1))
            lam_y = lam.reshape((-1, 1))

            x = x1 * lam_x + x2 * (1.0 - lam_x)
            y = y1 * lam_y + y2 * (1.0 - lam_y)
            yield x, y


# ========== Khối tiện ích ==========
def ConvBNReLU(x, filters, k=3, s=1, wd=WD):
    x = layers.Conv2D(filters, k, strides=s, padding="same",
                      use_bias=False, kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

# ========== Model ==========
def build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=5,
                wd=WD, drop_rate=DROP):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = ConvBNReLU(inputs, 32, 3, 1, wd)
    x = ConvBNReLU(x,      32, 3, 1, wd)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # Block 2
    x = ConvBNReLU(x, 64, 3, 1, wd)
    x = ConvBNReLU(x, 64, 3, 1, wd)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    x = layers.SpatialDropout2D(0.15)(x)    # chống overfit theo kênh

    # Block 3
    x = ConvBNReLU(x, 128, 3, 1, wd)
    x = ConvBNReLU(x, 128, 3, 1, wd)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # Block 4 (7th conv)
    x = ConvBNReLU(x, 256, 3, 1, wd)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    x = layers.SpatialDropout2D(0.15)(x)

    # >>> Thay Flatten bằng GAP để giảm tham số <<<
    x = layers.GlobalAveragePooling2D()(x)

    # FC nhỏ gọn + L2 + Dropout
    x = layers.Dense(256, use_bias=False, kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(drop_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=regularizers.l2(wd))(x)
    return models.Model(inputs, outputs, name="YogaPoseCNN13_anti_overfit")

# ========== Train/Eval ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--out_dir",   type=str, default="outputs_rgb")
    parser.add_argument("--epochs",    type=int, default=EPOCHS)
    parser.add_argument("--batch",     type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",        type=float, default=LR)
    parser.add_argument("--wd",        type=float, default=WD)
    parser.add_argument("--dropout",   type=float, default=DROP)
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTH)
    parser.add_argument("--mixup_alpha", type=float, default=MIXUP_ALPHA)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            try: tf.config.experimental.set_memory_growth(g, True)
            except: pass

    # Data
    train_gen, val_gen = build_datasets(args.data_root, IMG_SIZE, args.batch)
    num_classes = train_gen.num_classes
    class_names = list(train_gen.class_indices.keys())

    # MixUp wrapper
    train_steps = max(1, train_gen.samples // args.batch)  # chỉ full batch
    val_steps   = math.ceil(val_gen.samples / args.batch)   # val vẫn giữ nguyên
    train_gen_mix = mixup_generator(train_gen, alpha=args.mixup_alpha)

    # Model
    model = build_model(num_classes=num_classes, wd=args.wd, drop_rate=args.dropout)
    opt = optimizers.Adam(learning_rate=args.lr)

    # >>> Label smoothing để giảm tự tin quá mức <<<
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)

    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
    model.summary()

    # Callbacks: theo dõi val_loss để ngăn overfit
    ckpt = callbacks.ModelCheckpoint(
        os.path.join(args.out_dir, "best_model_tf.h5"),
        monitor="val_loss", mode="min", save_best_only=True
    )
    early = callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                    patience=8, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)

    history = model.fit(
        train_gen_mix,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=[ckpt, early, rlrop]
    )

    # Đánh giá
    val_gen.reset()
    probs = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = val_gen.classes

    # In báo cáo
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification report:\n")
    print(tf.keras.utils.to_categorical(y_true).shape)  # sanity check nhỏ
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("\nConfusion matrix:\n")
    print(confusion_matrix(y_true, y_pred))

    # Lưu class names
    with open(os.path.join(args.out_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
