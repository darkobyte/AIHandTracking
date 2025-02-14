import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import cv2
import numpy as np
import model_compression_toolkit as mct
from blazepalm import BlazePalm
from model_compression_toolkit.core import QuantizationErrorMethod


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


def main():
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

    # Save quantized model
    torch.save(quantized_model.state_dict(), "blazepalm_quantized.pth")
    print("Quantized model saved as 'blazepalm_quantized.pth'")

    # Export to ONNX
    mct.exporter.pytorch_export_model(quantized_model, "./blazepalm_quantized.onnx", repr_dataset=representative_data_gen)
    #export_onnx(quantized_model, "./blazepalm_quantized.onnx")

    # Print quantization info
    print("\nQuantization Info:")
    print(quantization_info)


if __name__ == "__main__":
    main()
