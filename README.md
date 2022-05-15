# Measure Execution Time of Audio Spectrogram Transformer Model
This project is to measure the execution time of Audio Spectrogram Transformer (AST) model
The source code of AST model can be found here
https://github.com/YuanGongND/ast

To find onnx model, pth model and openvino model, please go here
https://drive.google.com/drive/folders/1SLywEZQ6U6Dp1AhfZBzPncBT4ibHizw9?usp=sharing

1. About how to redo experiment of AST model with ESC 50 dataset, how to convert to .pth model to ONNX model, compare the execution time between pth model and ONNX model, you can find these points in Audio Spectrogram Transformer.pdf

2. How to convert from ONNX model to OpenVINO model
- Install openvino
  + $ pip install openvino-dev[pytorch,onnx]
- Generate openvino xml model. After running this comment, openvino will generate a .bin, .mapping and .xml openvino model. Note that in openvino 2021 (verison 10), you have to indicate a specific batch size in input_shape. From openvino 2022 (verison 11), you can set batch size to -1, then input various batch sizes
  + $ mo --input_model best_audio_model.onnx --input_shape "[5, 512, 128]" --data_type FP16
- Run .xml openvino model
  + $ python3 run_openvino.python
