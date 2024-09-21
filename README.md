# Fork of [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)

Differences between original repository and fork:

* Compatibility with PyTorch >=2.4. (🔥)
* Original pretrained models and converted ONNX models from GitHub [releases page](https://github.com/clibdev/Pytorch_Retinaface/releases). (🔥)
* Installation with [requirements.txt](requirements.txt) file.
* The [wider_val.txt](data/widerface/val/wider_val.txt) file for WIDERFace evaluation.
* Model is used for inference by default by setting pretrain to False in the [config.py](data/config.py) file.
* Minor modifications in the [detect.py](detect.py) and [convert_to_onnx.py](convert_to_onnx.py) file.
* The following deprecations has been fixed:
  * UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future.  
  * FutureWarning: 'torch.onnx._export' is deprecated in version 1.12.0 and will be removed in 2.0.
  * DeprecationWarning: 'np.float' is a deprecated alias for builtin 'float'.
  * FutureWarning: You are using 'torch.load' with 'weights_only=False'.
  * FutureWarning: Cython directive 'language_level' not set.
  * Cython Warning: Using deprecated NumPy API.

# Installation

```shell
pip install -r requirements.txt
```

# Pretrained models

* Download links:

| Name                      | Model Size (MB) | Link                                                                                                                                                                                                                             | SHA-256                                                                                                                              |
|---------------------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| RetinaFace ResNet-50      | 104.4<br>104.1  | [PyTorch](https://github.com/clibdev/Pytorch_Retinaface/releases/latest/download/retinaface-resnet-50.pth)<br>[ONNX](https://github.com/clibdev/Pytorch_Retinaface/releases/latest/download/retinaface-resnet-50.onnx)           | 6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d<br>a903411b598d92ccc6af5660d875f38783a93031be13643ed4c87851ec165898 |
| RetinaFace MobileNet-0.25 | 1.7<br>1.6      | [PyTorch](https://github.com/clibdev/Pytorch_Retinaface/releases/latest/download/retinaface-mobilenet-0.25.pth)<br>[ONNX](https://github.com/clibdev/Pytorch_Retinaface/releases/latest/download/retinaface-mobilenet-0.25.onnx) | 2979b33ffafda5d74b6948cd7a5b9a7a62f62b949cef24e95fd15d2883a65220<br>4aa128919a621c913b3bf78befa33eaaba086a7a7e142cdb739839a9340f2420 |

* Evaluation results on WIDERFace dataset:

| Name                      | Easy  | Medium | Hard  |
|---------------------------|-------|--------|-------|
| RetinaFace ResNet-50      | 95.48 | 94.04  | 84.43 |
| RetinaFace MobileNet-0.25 | 90.70 | 88.16  | 73.82 |

# Inference

```shell
python detect.py --trained_model weights/retinaface-resnet-50.pth --network resnet50 --image_path curve/test.jpg
python detect.py --trained_model weights/retinaface-mobilenet-0.25.pth --network mobile0.25 --image_path curve/test.jpg
```

# WIDERFace evaluation

* Download WIDERFace [validation dataset](https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view).
* Move dataset to `data/widerface/val` directory.

```shell
python test_widerface.py --trained_model weights/retinaface-mobilenet-0.25.pth --network mobile0.25 --dataset_folder data/widerface/val/images/
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
python convert_to_onnx.py --trained_model weights/retinaface-resnet-50.pth --network resnet50
python convert_to_onnx.py --trained_model weights/retinaface-mobilenet-0.25.pth --network mobile0.25
```
