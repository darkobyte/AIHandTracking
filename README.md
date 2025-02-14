# AI Cam Shit

## Requirements

### Optional using VENV
1. ```python -m venv venv```
2. ```source /venv/bin/activate```

1. First, install PyTorch following the instructions at: https://pytorch.org/get-started/locally/
2. ```pip install model_compression_toolkit```
3. ```pip install imx500-converter[pt]```

## Usage

1. Capture hand images first: ```python capture_hands.py```
2. Run Demo.py ```python demo.py```

### onnx to model
3. imxconv-pt -i .\quantized_models\blazepalm_quantized.onnx -o .\model

### model to pi cam (ONLY ON RASP)
4. sudo apt install imx500-tools
5. imx500-package -i <path to packerOut.zip> -o <output folder>

## Models ported so far
1. Face detector (BlazeFace)
1. Face landmarks
1. Palm detector
1. Hand landmarks