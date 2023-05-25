# GroundingDION

## Introduction


GroundingDION is a Open-Set object detection model. We reproduced the model of the [paper](https://arxiv.org/abs/2303.05499).


## Prepare
1. install
```
python setup.py install
```
2. Download the model weights file
```
wget https://bj.bcebos.com/v1/paddledet/models/groundingdino_swint_ogc.pdparams
```


## Demo
```bash
CUDA_VISIBLE_DEVICES={GPU ID} python demo/inference_on_a_image.py \
-c ppgroundingdino/config/GroundingDINO_SwinT_OGC.py \
-p groundingdino_swint_ogc.pdparams \
-i image_you_want_to_detect.jpg \
-o "dir you want to save the output" \
-t "Detect Cat"
 [--cpu-only] # open it for cpu mode
```


## Citations
```
@inproceedings{ShilongLiu2023GroundingDM,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author={Shilong Liu and Zhaoyang Zeng and Tianhe Ren and Feng Li and Hao Zhang and Jie Yang and Chunyuan Li and Jianwei Yang and Hang Su and Jun Zhu and Lei Zhang},
  year={2023}
}
```
