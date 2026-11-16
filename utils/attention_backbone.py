"""
Multi-Scale Attention-Augmented Network for Knee OA KL Severity Grading
========================================================================
Inspired by:
  Zhang et al. (Frontiers in Medicine, June 2026):
  "Automated Kellgren-Lawrence grading of knee osteoarthritis using a
   multi-scale attention-based deep learning framework"

Architecture Overview:
  1. Base Backbone: EfficientNet-B0 or EfficientNet-B2 (pre-trained on ImageNet).
  2. Feature Pyramid Network (FPN): Extracts multi-resolution representations from
     low-level (osteophyte margins), mid-level (joint space narrowing), and high-level
     (macro bone deformity) layers.
  3. Channel Attention (Squeeze-and-Excitation): Recalibrates channel importance.
  4. Spatial Self-Attention: Captures long-range bilateral dependencies across the joint.
  5. Ordinal Head: Consistent Rank Logits (CORAL) or Earth Mover's Distance (EMD) Softmax.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from .ordinal_loss import CoralOrdinalLayer
from .stn_layer import SpatialTransformer2D


# =====================================================================
# 1. ATTENTION MODULES
# =====================================================================

@tf.keras.utils.register_keras_serializable(package="CustomAttention")
class SqueezeAndExcitationBlock(layers.Layer):
    """
    Squeeze-and-Excitation (SE) Channel Attention Block.
    Models inter-channel dependencies by adaptively weighting feature channels.
    """
    def __init__(self, channels, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.reduced_channels = max(channels // reduction, 8)

    def build(self, input_shape):
        self.global_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = layers.Dense(self.reduced_channels, activation="relu", use_bias=False)
        self.fc2 = layers.Dense(self.channels, activation="sigmoid", use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (B, H, W, C)
        scale = self.global_pool(inputs)  # (B, 1, 1, C)
        scale = self.fc1(scale)           # (B, 1, 1, C // r)
        scale = self.fc2(scale)           # (B, 1, 1, C)
        return inputs * scale             # Channel-weighted output

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "reduction": self.reduction
        })
        return config


@tf.keras.utils.register_keras_serializable(package="CustomAttention")
class SpatialSelfAttentionBlock(layers.Layer):
    """
    Spatial Self-Attention Block for 2D Feature Maps.
    Allows features in the medial compartment to attend to the lateral compartment,
    capturing asymmetric joint-space narrowing across the tibial plateau.
    """
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        inter_channels = max(self.channels // 8, 8)
        self.query_conv = layers.Conv2D(inter_channels, kernel_size=1, padding="same")
        self.key_conv = layers.Conv2D(inter_channels, kernel_size=1, padding="same")
        self.value_conv = layers.Conv2D(self.channels, kernel_size=1, padding="same")
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1,),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (B, H, W, C)
        batch = tf.shape(inputs)[0]
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]
        hw = h * w

        # Query: (B, HW, C_inter)
        proj_query = self.query_conv(inputs)
        proj_query = tf.reshape(proj_query, [batch, hw, -1])

        # Key: (B, C_inter, HW)
        proj_key = self.key_conv(inputs)
        proj_key = tf.reshape(proj_key, [batch, hw, -1])
        proj_key = tf.transpose(proj_key, [0, 2, 1])

        # Energy matrix: (B, HW, HW)
        energy = tf.matmul(proj_query, proj_key)
        scale = tf.math.rsqrt(tf.cast(tf.shape(proj_query)[-1], tf.float32))
        attention = tf.nn.softmax(energy * scale, axis=-1)

        # Value: (B, HW, C)
        proj_value = self.value_conv(inputs)
        proj_value = tf.reshape(proj_value, [batch, hw, -1])

        # Output: (B, HW, C) -> (B, H, W, C)
        out = tf.matmul(attention, proj_value)
        out = tf.reshape(out, [batch, h, w, self.channels])

        # Residual connection scaled by learnable gamma
        return inputs + self.gamma * out

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


# =====================================================================
# 2. MULTI-SCALE ATTENTION BACKBONE FACTORY
# =====================================================================

def build_multiscale_attention_model(
    backbone_type="b2",
    input_shape=(224, 224, 3),
    pyramid_channels=128,
    num_classes=5,
    head_type="coral",
    dropout_rate=0.3,
    pretrained=True,
    use_stn=True
):
    """
    Builds the full end-to-end Multi-Scale Attention-Augmented Network with STN joint localization.

    Args:
        backbone_type: "b2" for EfficientNet-B2 (as in Zhang et al., ~9.8M params)
                       "b0" for EfficientNet-B0 (lighter, ~6.8M params)
        input_shape: Input resolution tuple, typically (224, 224, 3)
        pyramid_channels: Common channel depth across FPN levels (default 128)
        num_classes: Number of ordinal severity grades (5: Grades 0 to 4)
        head_type: "coral" for Consistent Rank Logits (4 monotonic sigmoids)
                   "softmax" for standard probability distribution (compatible with EMD loss)
        dropout_rate: Dropout rate for dense classification head
        pretrained: Whether to initialize backbone with ImageNet weights
        use_stn: Whether to prepend Spatial Transformer Network (STN) for differentiable joint localization

    Returns:
        tf.keras.Model
    """
    weights = "imagenet" if pretrained else None
    
    # 1. Base Backbone
    if backbone_type.lower() == "b0":
        base_backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=input_shape
        )
    else:
        # Default: EfficientNet-B2
        base_backbone = tf.keras.applications.EfficientNetB2(
            include_top=False,
            weights=weights,
            input_shape=input_shape
        )

    # Standardized stride 8, 16, 32 intermediate layer tap points
    # low-level: stride 8 (28x28 for 224 input)
    low_layer_name = "block3b_add"
    # mid-level: stride 16 (14x14 for 224 input)
    mid_layer_name = "block5c_add"
    # high-level: stride 32 (7x7 for 224 input)
    high_layer_name = "top_activation"

    # Multi-scale feature extractor sub-model
    low_feat_t = base_backbone.get_layer(low_layer_name).output
    mid_feat_t = base_backbone.get_layer(mid_layer_name).output
    high_feat_t = base_backbone.get_layer(high_layer_name).output
    
    extractor = Model(
        inputs=base_backbone.input,
        outputs=[low_feat_t, mid_feat_t, high_feat_t],
        name=f"{backbone_type.lower()}_feature_extractor"
    )

    # Raw Radiograph Input
    inputs = layers.Input(shape=input_shape, name="radiograph_input")

    # STN Differentiable Joint Space Alignment
    if use_stn:
        stn_layer = SpatialTransformer2D(
            output_size=(input_shape[0], input_shape[1]),
            name="stn_joint_aligner"
        )
        aligned_inputs = stn_layer(inputs)
        low_feat, mid_feat, high_feat = extractor(aligned_inputs)
    else:
        low_feat, mid_feat, high_feat = extractor(inputs)

    # 2. Feature Pyramid Network (FPN) Projections
    p_high = layers.Conv2D(pyramid_channels, kernel_size=1, padding="same", name="fpn_c_high")(high_feat)
    p_mid = layers.Conv2D(pyramid_channels, kernel_size=1, padding="same", name="fpn_c_mid")(mid_feat)
    p_low = layers.Conv2D(pyramid_channels, kernel_size=1, padding="same", name="fpn_c_low")(low_feat)

    # Top-Down Pathway
    # Upsample high to mid
    p_high_up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="up_high_to_mid")(p_high)
    p_mid_fused = layers.Add(name="fuse_mid")([p_mid, p_high_up])

    # Upsample mid to low
    p_mid_up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="up_mid_to_low")(p_mid_fused)
    p_low_fused = layers.Add(name="fuse_low")([p_low, p_mid_up])

    # 3. Attention Modules on Fused Pyramid Maps
    # High-level: SE Channel Attention + Spatial Self-Attention
    p_high_attn = SqueezeAndExcitationBlock(pyramid_channels, name="se_high")(p_high)
    p_high_attn = SpatialSelfAttentionBlock(pyramid_channels, name="self_attn_high")(p_high_attn)

    # Mid-level: SE Channel Attention
    p_mid_attn = SqueezeAndExcitationBlock(pyramid_channels, name="se_mid")(p_mid_fused)

    # Low-level: SE Channel Attention (preserves osteophytic boundary sharpness)
    p_low_attn = SqueezeAndExcitationBlock(pyramid_channels, name="se_low")(p_low_fused)

    # 4. Multi-Scale Pooling & Fusion
    pool_high = layers.GlobalAveragePooling2D(name="pool_high")(p_high_attn)
    pool_mid = layers.GlobalAveragePooling2D(name="pool_mid")(p_mid_attn)
    pool_low = layers.GlobalAveragePooling2D(name="pool_low")(p_low_attn)

    fused_features = layers.Concatenate(name="multiscale_concat")([pool_high, pool_mid, pool_low])

    # 5. Classifier Head
    x = layers.Dense(256, activation="relu", name="head_dense1")(fused_features)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout1")(x)

    x = layers.Dense(128, activation="relu", name="head_dense2")(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(dropout_rate * 0.7, name="head_dropout2")(x)

    # 6. Final Task Head
    if head_type.lower() == "coral":
        # 4 Monotonic Threshold Sigmoids
        outputs = CoralOrdinalLayer(num_classes=num_classes, name="coral_ordinal_output")(x)
    else:
        # Standard 5-class nominal Softmax (compatible with EMD loss)
        outputs = layers.Dense(num_classes, activation="softmax", name="softmax_output")(x)

    stn_tag = "STN_" if use_stn else ""
    model = Model(inputs=inputs, outputs=outputs, name=f"MultiScale_Attention_{stn_tag}{backbone_type.upper()}_{head_type.upper()}")
    return model
