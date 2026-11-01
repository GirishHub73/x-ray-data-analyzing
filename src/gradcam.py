import cv2
import numpy as np
import tensorflow as tf
def prepare_image(image, image_size=224):
    grayscale = image.convert("L")
    grayscale_array = np.asarray(grayscale, dtype=np.float32)
    rgb_array = np.repeat(
        grayscale_array[..., np.newaxis], 3, axis=2
    )
    resized = tf.image.resize(
        rgb_array, [image_size, image_size]
    ).numpy()
    display_image = np.uint8(np.clip(resized, 0, 255))
    model_input = np.expand_dims(resized, axis=0)
    return model_input, display_image
def make_gradcam_heatmap(model, image_array):
    base_model = model.get_layer("efficientnetb0")
    convolution_layers = [
        layer for layer in reversed(base_model.layers)
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    if not convolution_layers:
        raise ValueError("No convolutional layer was found.")
    last_convolution_layer = convolution_layers[0]
    grad_model = tf.keras.models.Model(
        model.inputs,
        [last_convolution_layer.output, model.output],
    )
    with tf.GradientTape() as tape:
        convolution_output, prediction = grad_model(image_array)
        target_output = prediction[:, 0]
    gradients = tape.gradient(target_output, convolution_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    convolution_output = convolution_output[0]
    heatmap = tf.reduce_sum(
        convolution_output * pooled_gradients,
        axis=-1,
    )
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )
    return heatmap.numpy()
def create_overlay(display_image, heatmap):
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )
    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB,
    )
    return cv2.addWeighted(
        display_image,
        0.60,
        colored_heatmap,
        0.40,
        0,
    )