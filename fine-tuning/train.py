import json
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import dataloader as dl
import load_model


def setup_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


@dataclass
class TrainConfig:
    image_dir: str = "./dataset/train"
    label_path: str = "./dataset/train/_annotation.wireframe.json"
    val_image_dir: str = "./dataset/val"
    val_label_path: str = "./dataset/val/_annotation.wireframe.json"
    ckpt_dir: str = "./ckpt_models/M-LSD_512_large"
    save_dir: str = "./checkpoints/finetune"

    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.001

    warmup_epochs: int = 5
    decay_start_epoch: int = 70
    eval_every: int = 1

    loss_weight_cls: float = 1.0
    loss_weight_reg: float = 1.0
    augment: bool = True


def load_labels(label_path: str, image_dir: str) -> Tuple[List[str], List[List[List[float]]]]:
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Label json not found: {label_path}")

    with open(label_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    image_paths = []
    lines_list = []
    for item in labels:
        filename = item.get("filename")
        lines = item.get("lines", [])
        if not filename:
            continue
        image_paths.append(os.path.join(image_dir, filename))
        lines_list.append(lines)

    if not image_paths:
        raise ValueError("No labels found in label json.")

    return image_paths, lines_list


def get_lr(epoch: int, config: TrainConfig) -> float:
    if epoch < config.warmup_epochs:
        return config.learning_rate * (epoch / max(1, config.warmup_epochs))
    if epoch < config.decay_start_epoch:
        return config.learning_rate

    progress = (epoch - config.decay_start_epoch) / max(1, config.epochs - config.decay_start_epoch)
    return 0.5 * config.learning_rate * (1.0 + math.cos(math.pi * progress))


def _masked_huber(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    mask: tf.Tensor,
    huber: tf.keras.losses.Huber,
) -> tf.Tensor:
    true_vals = tf.boolean_mask(y_true, mask)
    pred_vals = tf.boolean_mask(y_pred, mask)
    return tf.cond(
        tf.size(true_vals) > 0,
        lambda: tf.reduce_mean(huber(true_vals, pred_vals)),
        lambda: tf.constant(0.0, dtype=tf.float32),
    )


def compute_loss(y_true: tf.Tensor, y_pred: tf.Tensor, config: TrainConfig) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    cls_idx = [0, 7, 14, 15]
    y_true_cls = tf.gather(y_true, cls_idx, axis=-1)
    y_pred_cls = tf.gather(y_pred, cls_idx, axis=-1)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    cls_loss = tf.reduce_mean(bce(y_true_cls, y_pred_cls))

    huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    mask_a = y_true[..., 0] > 0.0
    mask_b = y_true[..., 7] > 0.0

    reg_loss_a = _masked_huber(y_true[..., 1:7], y_pred[..., 1:7], mask_a, huber)
    reg_loss_b = _masked_huber(y_true[..., 8:14], y_pred[..., 8:14], mask_b, huber)
    reg_loss = reg_loss_a + reg_loss_b

    total_loss = (cls_loss * config.loss_weight_cls) + (reg_loss * config.loss_weight_reg)
    return total_loss, cls_loss, reg_loss


def main() -> None:
    setup_gpu()
    config = TrainConfig()

    model_cfg = load_model.infer_config_from_path(config.ckpt_dir)
    model_cfg.batch_size = config.batch_size
    model = load_model.load_pretrained_model(model_cfg, config.ckpt_dir, return_train_map=True)

    image_paths, lines_list = load_labels(config.label_path, config.image_dir)
    dataset = dl.build_dataloader(
        image_paths=image_paths,
        lines_list=lines_list,
        batch_size=config.batch_size,
        target_size=model_cfg.input_size,
        augment=config.augment,
    )

    val_dataset = None
    if os.path.isfile(config.val_label_path):
        val_image_paths, val_lines_list = load_labels(config.val_label_path, config.val_image_dir)
        val_dataset = dl.build_dataloader(
            image_paths=val_image_paths,
            lines_list=val_lines_list,
            batch_size=config.batch_size,
            target_size=model_cfg.input_size,
            augment=False,
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0)
    if tf.keras.mixed_precision.global_policy().compute_dtype == "float16":
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    ckpt = tf.train.Checkpoint(step=tf.Variable(0, name="step"), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, config.save_dir, max_to_keep=5)

    @tf.function
    def train_step(images: tf.Tensor, targets: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            preds = model(images, training=True)
            total_loss, cls_loss, reg_loss = compute_loss(targets, preds, config)

            if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                scaled_loss = optimizer.get_scaled_loss(total_loss)
            else:
                scaled_loss = total_loss

        if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
            grads = optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(scaled_loss, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, cls_loss, reg_loss

    @tf.function
    def eval_step(images: tf.Tensor, targets: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        preds = model(images, training=False)
        total_loss, cls_loss, reg_loss = compute_loss(targets, preds, config)
        return total_loss, cls_loss, reg_loss

    for epoch in range(config.epochs):
        lr = get_lr(epoch, config)
        optimizer.learning_rate.assign(lr)

        total_meter = tf.keras.metrics.Mean()
        cls_meter = tf.keras.metrics.Mean()
        reg_meter = tf.keras.metrics.Mean()

        for images, targets in dataset:
            total_loss, cls_loss, reg_loss = train_step(images, targets)
            total_meter.update_state(total_loss)
            cls_meter.update_state(cls_loss)
            reg_meter.update_state(reg_loss)

        log = (
            f"Epoch {epoch + 1}/{config.epochs} | lr={lr:.6f} | "
            f"loss={total_meter.result():.4f} | "
            f"cls={cls_meter.result():.4f} | reg={reg_meter.result():.4f}"
        )

        if val_dataset is not None and (epoch + 1) % config.eval_every == 0:
            val_total = tf.keras.metrics.Mean()
            val_cls = tf.keras.metrics.Mean()
            val_reg = tf.keras.metrics.Mean()
            for images, targets in val_dataset:
                total_loss, cls_loss, reg_loss = eval_step(images, targets)
                val_total.update_state(total_loss)
                val_cls.update_state(cls_loss)
                val_reg.update_state(reg_loss)
            log += (
                f" | val_loss={val_total.result():.4f}"
                f" | val_cls={val_cls.result():.4f}"
                f" | val_reg={val_reg.result():.4f}"
            )

        print(log)

        ckpt.step.assign_add(1)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.epochs:
            manager.save()


if __name__ == "__main__":
    main()
