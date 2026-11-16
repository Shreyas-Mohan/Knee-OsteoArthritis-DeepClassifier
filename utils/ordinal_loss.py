"""
Ordinal Loss Functions and Layers for Progressive Disease Grading
==================================================================
This module implements mathematically strict ordinal regression architectures and losses
tailored for progressive disease severity scoring (Kellgren-Lawrence Grades 0-4).

Supported approaches:
1. CORAL (Consistent Rank Logits - Cao et al.):
   Shared weight projection with task-specific threshold biases.
   Mathematically guarantees rank consistency across binary splits.
2. Earth Mover's Distance (EMD) / Wasserstein Distance Loss:
   Penalizes probabilistic divergence quadratically by distance across ordinal grades.
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# =====================================================================
# 1. LABEL CONVERSION UTILITIES
# =====================================================================

def grade_to_ordinal_vector(grade, num_classes=5):
    """
    Converts integer KL grade (0..num_classes-1) to (num_classes-1) binary targets.
    Grade 0 -> [0, 0, 0, 0]
    Grade 1 -> [1, 0, 0, 0]
    Grade 2 -> [1, 1, 0, 0]
    Grade 3 -> [1, 1, 1, 0]
    Grade 4 -> [1, 1, 1, 1]
    """
    num_thresholds = num_classes - 1
    binary_targets = np.zeros(num_thresholds, dtype=np.float32)
    if grade > 0:
        binary_targets[:min(grade, num_thresholds)] = 1.0
    return binary_targets


def batch_grades_to_ordinal(grades, num_classes=5):
    """Vectorized conversion for batch of integer grades."""
    num_thresholds = num_classes - 1
    thresholds = tf.range(1, num_classes, dtype=tf.float32)
    grades_col = tf.cast(tf.reshape(grades, [-1, 1]), tf.float32)
    return tf.cast(grades_col >= thresholds, tf.float32)


def ordinal_probs_to_grade(probs, threshold=0.5):
    """
    Converts (batch, num_classes-1) predicted cumulative probabilities
    into integer KL grades [0..num_classes-1] by counting exceedances.
    """
    if isinstance(probs, tf.Tensor):
        probs = probs.numpy()
    return np.sum(probs > threshold, axis=-1).astype(int)


def ordinal_probs_to_class_distribution(probs):
    """
    Derives the complete discrete probability distribution P(Y = k) for k in [0..4]
    from cumulative threshold probabilities P(Y >= 1), P(Y >= 2), P(Y >= 3), P(Y >= 4).

    Returns:
        Array of shape (batch, num_classes) where sum(P(Y=k)) == 1.
    """
    if isinstance(probs, tf.Tensor):
        probs = probs.numpy()
    if probs.ndim == 1:
        probs = np.expand_dims(probs, axis=0)
        squeeze = True
    else:
        squeeze = False

    # Enforce empirical monotonicity P(Y >= k) <= P(Y >= k-1)
    mono_probs = np.copy(probs)
    for c in range(1, mono_probs.shape[1]):
        mono_probs[:, c] = np.minimum(mono_probs[:, c], mono_probs[:, c - 1])

    batch_size = mono_probs.shape[0]
    num_thresholds = mono_probs.shape[1]
    num_classes = num_thresholds + 1
    dist = np.zeros((batch_size, num_classes), dtype=np.float32)

    # P(Y = 0) = 1 - P(Y >= 1)
    dist[:, 0] = np.clip(1.0 - mono_probs[:, 0], 0.0, 1.0)

    # P(Y = k) = P(Y >= k) - P(Y >= k+1)
    for k in range(1, num_thresholds):
        dist[:, k] = np.clip(mono_probs[:, k - 1] - mono_probs[:, k], 0.0, 1.0)

    # P(Y = K-1) = P(Y >= K-1)
    dist[:, num_classes - 1] = np.clip(mono_probs[:, -1], 0.0, 1.0)

    # Re-normalize to avoid numerical precision drift
    row_sums = dist.sum(axis=-1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    dist = dist / row_sums

    return dist[0] if squeeze else dist


# =====================================================================
# 2. CORAL ORDINAL LAYER & LOSS
# =====================================================================

@tf.keras.utils.register_keras_serializable(package="CustomOrdinal")
class CoralOrdinalLayer(layers.Layer):
    """
    CORAL (Consistent Rank Logits) Classification Head.
    
    Architecture:
      Inputs: Dense feature vector of dimension d.
      Weight: Shared weight vector W in R^(d x 1).
      Biases: K-1 distinct threshold biases [b_1, b_2, ..., b_{K-1}].
      
      Logit_k = W^T * x + b_k
      P(Y >= k) = sigmoid(Logit_k)
      
    Because W is shared across all thresholds, the relative rank order of logits
    is strictly governed by threshold offsets, preventing contradictory decision boundaries.
    """
    def __init__(self, num_classes=5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # Shared projection vector
        self.shared_kernel = self.add_weight(
            name="shared_kernel",
            shape=(input_dim, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        # Threshold-specific biases, initialized in descending order
        # to encourage b_1 > b_2 > b_3 > b_4 initially
        initial_biases = np.linspace(1.5, -1.5, self.num_thresholds, dtype=np.float32)
        self.threshold_biases = self.add_weight(
            name="threshold_biases",
            shape=(self.num_thresholds,),
            initializer=tf.constant_initializer(initial_biases),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, input_dim)
        # linear_proj: (batch, 1)
        linear_proj = tf.matmul(inputs, self.shared_kernel)
        # Broadcast biases: (batch, num_thresholds)
        logits = linear_proj + self.threshold_biases
        # Output cumulative probabilities
        return tf.sigmoid(logits)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


@tf.keras.utils.register_keras_serializable(package="CustomOrdinal")
class CoralLoss(tf.keras.losses.Loss):
    """
    Consistent Rank Logits Loss.
    Computes cross-entropy over all K-1 binary threshold indicators.
    """
    def __init__(self, class_weights=None, reduction="sum_over_batch_size", name="coral_loss"):
        super().__init__(reduction=reduction, name=name)
        if class_weights is not None:
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        else:
            self.class_weights = None

    def call(self, y_true, y_pred):
        # y_true: (batch, num_thresholds) binary flags [1, 1, 0, 0]
        # y_pred: (batch, num_thresholds) sigmoid probabilities
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        # Binary cross entropy across thresholds
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))

        if self.class_weights is not None:
            # y_true grade can be computed by summing threshold flags
            grade_indices = tf.cast(tf.reduce_sum(y_true, axis=-1), tf.int32)
            weights = tf.gather(self.class_weights, grade_indices)
            bce = bce * tf.expand_dims(weights, -1)

        # Mean over thresholds, reduction will handle batch dimension
        return tf.reduce_mean(bce, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "class_weights": self.class_weights.numpy().tolist() if self.class_weights is not None else None
        })
        return config


# =====================================================================
# 3. EARTH MOVER'S DISTANCE (WASSERSTEIN) ORDINAL LOSS
# =====================================================================

@tf.keras.utils.register_keras_serializable(package="CustomOrdinal")
class EarthMoversDistanceLoss(tf.keras.losses.Loss):
    """
    Squared Earth Mover's Distance (Wasserstein Distance) for 5-class nominal Softmax probabilities.
    Directly measures the distance between predicted and ground-truth CDFs:
        L_EMD = sum_{k=0}^{K-2} | CDF_pred(k) - CDF_true(k) |^2
    
    Penalizes errors proportionally to square of ordinal grade discrepancy.
    """
    def __init__(self, reduction="sum_over_batch_size", name="emd_loss"):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        # y_true: (batch, 5) one-hot encoded OR integer labels
        if tf.rank(y_true) == 1 or (tf.rank(y_true) == 2 and tf.shape(y_true)[-1] == 1):
            y_true = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), depth=tf.shape(y_pred)[-1])

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Cumulative distribution functions
        cdf_true = tf.cumsum(y_true, axis=-1)
        cdf_pred = tf.cumsum(y_pred, axis=-1)

        # Squared error between CDFs across the first K-1 points (last point is always 1)
        cdf_diff = tf.square(cdf_pred[:, :-1] - cdf_true[:, :-1])
        return tf.reduce_mean(cdf_diff, axis=-1)


# =====================================================================
# 4. ORDINAL EVALUATION METRICS
# =====================================================================

def compute_quadratic_weighted_kappa(y_true, y_pred_grades, num_classes=5):
    """
    Computes Cohen's Quadratic Weighted Kappa (QWK) on discrete grades.
    Standard gold-standard competition metric for medical ordinal grading.
    """
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y_true, y_pred_grades, weights="quadratic")


def compute_ordinal_mae(y_true, y_pred_grades):
    """Computes Mean Absolute Error between discrete integer grades."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred_grades))))

