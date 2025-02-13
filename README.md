# AI Cam Shit

## Requirements

First, install PyTorch following the instructions at:
https://pytorch.org/get-started/locally/

## Usage

1. Capture hand images first: ```python capture_hands.py```
2. Run Demo.py ```python demo.py```

### onnx to model
3. imxconv-pt -i .\blazepalm_quantized.onnx -o .\model

## Models ported so far
1. Face detector (BlazeFace)
1. Face landmarks
1. Palm detector
1. Hand landmarks