import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import cv2
import numpy as np
import model_compression_toolkit as mct
from blazepalm import BlazePalm
from model_compression_toolkit.core import QuantizationErrorMethod
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn.functional as F


def load_and_preprocess_images(image_dir="handpic"):
    """Load and preprocess images from handpic directory"""
    images = []
    for img_name in sorted(os.listdir(image_dir)):
        if img_name.endswith(".jpg"):

            # Read image
            img_path = os.path.join(image_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, (128, 128))  # Resize to 128x128
            img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
            img = img.transpose(2, 0, 1)  # Convert to NCHW format
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            images.append(img)
    return images


def representative_data_gen():
    """Generator function that yields batches of preprocessed images"""
    images = load_and_preprocess_images()
    for img in images:
        yield [img]


def export_onnx(model, save_path="./best.onnx"):
    """Export PyTorch model to ONNX format"""
    model = model.cpu()
    model.eval()
    dummy_input = torch.randn(1, 3, 128, 128)

    torch.onnx.export(
        model,  # Model being run
        dummy_input,  # Model input
        save_path,  # Where to save
        export_params=True,  # Store trained parameter weights
        opset_version=17,
        do_constant_folding=True,  # Execute constant folding
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model exported to ONNX format at: {save_path}")


def evaluate_models(original_model, quantized_model, test_data):
    """Compare original and quantized model outputs"""
    original_model.eval()
    quantized_model.eval()

    mse_scores = []
    mae_scores = []
    detection_matches = 0
    total_samples = 0

    with torch.no_grad():
        for img in test_data:
            img_tensor = torch.from_numpy(img)

            # Get predictions
            orig_output = original_model(img_tensor)
            quant_output = quantized_model(img_tensor)

            # Calculate MSE and MAE for regression outputs
            mse = mean_squared_error(
                orig_output[0].numpy().flatten(), quant_output[0].numpy().flatten()
            )
            mae = mean_absolute_error(
                orig_output[0].numpy().flatten(), quant_output[0].numpy().flatten()
            )

            mse_scores.append(mse)
            mae_scores.append(mae)

            # Compare detection results (assuming threshold of 0.5)
            orig_detected = (orig_output[1].numpy() > 0.5).astype(int)
            quant_detected = (quant_output[1].numpy() > 0.5).astype(int)
            detection_matches += np.sum(orig_detected == quant_detected)
            total_samples += orig_detected.size

    avg_mse = np.mean(mse_scores)
    avg_mae = np.mean(mae_scores)
    detection_accuracy = detection_matches / total_samples * 100

    return {
        "avg_mse": avg_mse,
        "avg_mae": avg_mae,
        "detection_accuracy": detection_accuracy,
    }


def main():
    os.makedirs("quantized_models", exist_ok=True)

    # Initialize model
    model = BlazePalm()
    model.load_weights("blazepalm.pth")
    model.load_anchors("anchors_palm.npy")
    model.eval()

    # Configure quantization settings
    core_config = mct.core.CoreConfig()

    target_platform_cap = mct.get_target_platform_capabilities(
        "pytorch", "imx500", target_platform_version="v1"
    )

    q_config = mct.core.QuantizationConfig(
        activation_error_method=QuantizationErrorMethod.MSE,
        weights_error_method=QuantizationErrorMethod.MSE,
        weights_bias_correction=True,
        shift_negative_activation_correction=True,
        z_threshold=16,
    )

    ptq_config = mct.core.CoreConfig(quantization_config=q_config)

    # Perform post-training quantization
    quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(
        model,
        representative_data_gen,
        core_config=core_config,
        target_platform_capabilities=target_platform_cap,
    )

    # Evaluate models
    print("\nEvaluating model accuracy...")
    test_data = load_and_preprocess_images()  # Using same data for testing
    metrics = evaluate_models(model, quantized_model, test_data)

    print("\nAccuracy Metrics:")
    print(f"Average MSE: {metrics['avg_mse']:.6f}")
    print(f"Average MAE: {metrics['avg_mae']:.6f}")
    print(f"Detection Accuracy: {metrics['detection_accuracy']:.2f}%")

    # Save quantized model
    save_path = os.path.join("quantized_models", "blazepalm_quantized.pth")
    torch.save(quantized_model.state_dict(), save_path)
    print(f"Quantized model saved as '{save_path}'")

    # Export to ONNX
    onnx_path = os.path.join("quantized_models", "blazepalm_quantized.onnx")
    mct.exporter.pytorch_export_model(
        quantized_model, onnx_path, repr_dataset=representative_data_gen
    )

    # Print quantization info
    print("\nQuantization Info:")
    print(quantization_info)


if __name__ == "__main__":
    main()
