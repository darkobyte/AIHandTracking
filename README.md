# AI Cam Shit

## Requirements

1. First, install PyTorch following the instructions at: https://pytorch.org/get-started/locally/
2. pip install model_compression_toolkit
3. pip install imx500-converter[pt]

## Usage

1. Capture hand images first: ```python capture_hands.py```
2. Run Demo.py ```python demo.py```

## Models ported so far
1. Face detector (BlazeFace)
1. Face landmarks
1. Palm detector
1. Hand landmarks