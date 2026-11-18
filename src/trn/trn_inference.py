
"""Inference for frozen SRN followed by masked TRN refinement."""

import numpy as np
import tensorflow as tf


def _load_trn_generator(path):
    """
    Load an older TRN model with Keras 3 compatibility.

    The saved model contains legacy BatchNormalization settings
    that are no longer accepted by Keras 3.
    """
    batch_norm = tf.keras.layers.BatchNormalization
    original_init = batch_norm.__init__

    def compatible_init(
        self,
        *args,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        **kwargs
    ):
        # Ignore legacy arguments that Keras 3 no longer supports.
        original_init(self, *args, **kwargs)

    batch_norm.__init__ = compatible_init

    try:
        generator = tf.keras.models.load_model(
            path,
            compile=False
        )
    finally:
        # Restore the original constructor after loading.
        batch_norm.__init__ = original_init

    return generator


def load_models(srn_path, generator_path):
    """Load the frozen SRN and the TRN generator."""

    srn = tf.keras.models.load_model(
        srn_path,
        compile=False
    )

    srn.trainable = False

    generator = _load_trn_generator(generator_path)

    return srn, generator


def refine(
    srn,
    generator,
    x,
    ground_truth=None,
    residual_scale=0.1
):
    """
    Run SRN followed by masked TRN refinement.

    x must contain normalized RGB and a binary mask:
    shape (batch, 128, 128, 4).
    Mask value 1 = missing region.
    Mask value 0 = known region.
    """

    x = tf.convert_to_tensor(x, tf.float32)

    if x.shape.rank == 3:
        x = x[None, ...]

    if x.shape[-1] != 4:
        raise ValueError(
            "Expected input shaped (batch, 128, 128, 4): "
            "RGB plus mask."
        )

    mask = x[..., 3:4]

    # Stage 1: Coarse structure reconstruction.
    srn_output = srn(x, training=False)

    input_rgb = x[..., :3]

    # Stage 2: TRN predicts a residual correction.
    trn_input = tf.concat(
        [srn_output, mask],
        axis=-1
    )

    residual = generator(
        trn_input,
        training=False
    )

    # Apply the residual correction.
    trn_candidate = tf.clip_by_value(
        srn_output + residual_scale * residual,
        0.0,
        1.0
    )

    # Preserve all known pixels from the original input.
    final_output = (
        input_rgb * (1.0 - mask)
        + trn_candidate * mask
    )

    result = {
        "masked_input": input_rgb.numpy(),
        "mask": mask.numpy(),
        "srn_output": srn_output.numpy(),
        "residual": residual.numpy(),
        "trn_candidate": trn_candidate.numpy(),
        "final_output": final_output.numpy()
    }

    if ground_truth is not None:
        result["ground_truth"] = np.asarray(ground_truth)

    return result
