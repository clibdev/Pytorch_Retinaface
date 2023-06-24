# Fork of [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)

Differences between original repository and fork:

* Compatibility with PyTorch >=2.0. (🔥)
* Original pretrained models and converted ONNX models from GitHub [releases page](https://github.com/clibdev/Pytorch_Retinaface/releases). (🔥)
* Installation with [requirements.txt](requirements.txt) file.
* The [wider_val.txt](data/widerface/val/wider_val.txt) file for WIDERFace evaluation.
* Model is used for inference by default by setting pretrain to False in the [config.py](data/config.py) file.
* Minor modifications in the [detect.py](detect.py) and [convert_to_onnx.py](convert_to_onnx.py) file.
* The following deprecations has been fixed:
  * UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future.  
  * FutureWarning: 'torch.onnx._export' is deprecated in version 1.12.0 and will be removed in 2.0.
  * DeprecationWarning: 'np.float' is a deprecated alias for builtin 'float'.
  * FutureWarning: Cython directive 'language_level' not set.
  * Cython Warning: Using deprecated NumPy API.

# Installation

```shell
pip install -r requirements.txt
```

# Pretrained models

| Name                                               | Easy  | Medium | Hard  | Link                                                                                                                                                                                                               |
|----------------------------------------------------|-------|--------|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Resnet50 backbone (same parameter with Mxnet)      | 94.82 | 93.84  | 89.60 | [PyTorch](https://github.com/clibdev/Pytorch_Retinaface/releases/latest/download/Resnet50_Final.pth), [ONNX](https://github.com/clibdev/Pytorch_Retinaface/releases/latest/download/Resnet50_Final.onnx)           |
| Resnet50 backbone (original image scale)           | 95.48 | 94.04  | 84.43 | [PyTorch](https://github.com/clibdev/Pytorch_Retinaface/releases/latest/download/Resnet50_Final.pth), [ONNX](https://github.com/clibdev/Pytorch_Retinaface/releases/latest/download/Resnet50_Final.onnx)           |
| Mobilenet0.25 backbone (same parameter with Mxnet) | 88.67 | 87.09  | 80.99 | [PyTorch](https://github.com/clibdev/Pytorch_Retinaface/releases/latest/download/mobilenet0.25_Final.pth), [ONNX](https://github.com/clibdev/Pytorch_Retinaface/releases/latest/download/mobilenet0.25_Final.onnx) |
| Mobilenet0.25 backbone (original image scale)      | 90.70 | 88.16  | 73.82 | [PyTorch](https://github.com/clibdev/Pytorch_Retinaface/releases/latest/download/mobilenet0.25_Final.pth), [ONNX](https://github.com/clibdev/Pytorch_Retinaface/releases/latest/download/mobilenet0.25_Final.onnx) |

# Inference

```shell
python detect.py --trained_model weights/Resnet50_Final.pth --network resnet50 --image_path curve/test.jpg
python detect.py --trained_model weights/mobilenet0.25_Final.pth --network mobile0.25 --image_path curve/test.jpg
```

# WIDERFace evaluation

* Download WIDERFace [validation dataset](https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view).
* Move dataset to `data/widerface/val` directory.

```shell
python test_widerface.py --trained_model weights/mobilenet0.25_Final.pth --network mobile0.25 --dataset_folder data/widerface/val/images/
```
```shell
cd widerface_evaluate
```
```shell
python setup.py build_ext --inplace
```
```shell
python evaluation.py
```

# Export to ONNX format

```shell
pip install onnx
```
```shell
python convert_to_onnx.py --trained_model weights/Resnet50_Final.pth --network resnet50
python convert_to_onnx.py --trained_model weights/mobilenet0.25_Final.pth --network mobile0.25
```
