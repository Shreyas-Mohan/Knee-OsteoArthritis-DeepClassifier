"""
Spatial Transformer Network (STN) for Differentiable Joint Localization
========================================================================
Implements a fully differentiable 2D Spatial Transformer Network (Jaderberg et al., 2015).
Used in knee radiography to automatically detect, crop, and align the tibiofemoral joint space
without requiring non-differentiable bounding boxes or external object detectors.

Components:
1. Localization Network: Predicts 6 affine transformation parameters [s_x, 0, t_x, 0, s_y, t_y]
2. Grid Generator: Generates normalized sampling grid in [-1, 1]
3. Bilinear Sampler: Differentiable sub-pixel bilinear interpolation
"""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="CustomSTN")
class SpatialTransformer2D(layers.Layer):
    """
    Differentiable Spatial Transformer Layer (2D Affine Alignment).
    """
    def __init__(self, output_size=(224, 224), **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size

    def build(self, input_shape):
        # Localization network: extracts spatial coordinates of joint space
        self.loc_net = tf.keras.Sequential([
            layers.Conv2D(32, kernel_size=5, strides=2, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(128, kernel_size=3, strides=2, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            # Output 6 parameters of affine matrix:
            # [[theta_11, theta_12, theta_13],
            #  [theta_21, theta_22, theta_23]]
            layers.Dense(
                6,
                kernel_initializer="zeros",
                bias_initializer=tf.constant_initializer([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            )
        ], name="stn_localization_net")
        super().build(input_shape)

    def _generate_grid(self, height, width):
        # Create normalized coordinate grid in [-1, 1]
        x = tf.linspace(-1.0, 1.0, width)
        y = tf.linspace(-1.0, 1.0, height)
        x_t, y_t = tf.meshgrid(x, y)
        x_t_flat = tf.reshape(x_t, [-1])
        y_t_flat = tf.reshape(y_t, [-1])
        ones = tf.ones_like(x_t_flat)
        grid = tf.stack([x_t_flat, y_t_flat, ones])  # (3, H*W)
        return grid

    def _bilinear_sampler(self, img, x, y):
        # img: (B, H, W, C)
        B = tf.shape(img)[0]
        H = tf.shape(img)[1]
        W = tf.shape(img)[2]
        C = tf.shape(img)[3]

        H_f = tf.cast(H, tf.float32)
        W_f = tf.cast(W, tf.float32)

        # Rescale coordinates from [-1, 1] to pixel space [0, W-1] and [0, H-1]
        x = (x + 1.0) * (W_f - 1.0) / 2.0
        y = (y + 1.0) * (H_f - 1.0) / 2.0

        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, 0, W - 1)
        x1 = tf.clip_by_value(x1, 0, W - 1)
        y0 = tf.clip_by_value(y0, 0, H - 1)
        y1 = tf.clip_by_value(y1, 0, H - 1)

        # Batch offsets for indexing
        b_idx = tf.range(B)
        b_idx = tf.reshape(b_idx, [-1, 1])
        b_idx = tf.tile(b_idx, [1, tf.shape(x)[1]])

        def get_pixel(y_idx, x_idx):
            indices = tf.stack([b_idx, y_idx, x_idx], axis=-1)
            return tf.gather_nd(img, indices)

        Ia = get_pixel(y0, x0)
        Ib = get_pixel(y1, x0)
        Ic = get_pixel(y0, x1)
        Id = get_pixel(y1, x1)

        x0_f = tf.cast(x0, tf.float32)
        x1_f = tf.cast(x1, tf.float32)
        y0_f = tf.cast(y0, tf.float32)
        y1_f = tf.cast(y1, tf.float32)

        wa = (x1_f - x) * (y1_f - y)
        wb = (x1_f - x) * (y - y0_f)
        wc = (x - x0_f) * (y1_f - y)
        wd = (x - x0_f) * (y - y0_f)

        wa = tf.expand_dims(wa, -1)
        wb = tf.expand_dims(wb, -1)
        wc = tf.expand_dims(wc, -1)
        wd = tf.expand_dims(wd, -1)

        out = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return out

    def call(self, inputs):
        # inputs: (B, H, W, C)
        theta = self.loc_net(inputs)
        theta = tf.reshape(theta, [-1, 2, 3])  # (B, 2, 3)

        out_h, out_w = self.output_size
        grid = self._generate_grid(out_h, out_w)  # (3, out_H * out_W)
        grid = tf.expand_dims(grid, 0)
        grid = tf.tile(grid, [tf.shape(inputs)[0], 1, 1])  # (B, 3, out_H * out_W)

        # Transform target grid into source sampling grid
        sampling_grid = tf.matmul(theta, grid)  # (B, 2, out_H * out_W)
        x_s = sampling_grid[:, 0, :]
        y_s = sampling_grid[:, 1, :]

        # Differentiable Bilinear Resampling
        sampled = self._bilinear_sampler(inputs, x_s, y_s)
        out = tf.reshape(sampled, [-1, out_h, out_w, tf.shape(inputs)[-1]])
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"output_size": self.output_size})
        return config

