# <div style="text-align: center;">BiFusionNet: A Lightweight Model for Detecting Red Turpentine Beetle Infestation in Pine Trees </div>

## PDT Dataset
You can download the dataset used for BiFusionNet training from the following link
  **Hugging Face:** [PDT dataset v2 (Improve the quality 2024.10.4)](https://huggingface.co/datasets/qwer0213/PDT_dataset/tree/main)
```
@inproceedings{zhou2024pdt,
  title={PDT: Uav Target Detection Dataset for Pests and Diseases Tree},
  author={Zhou, Mingle and Xing, Rui and Han, Delong and Qi, Zhiyong and Li, Gang},
  booktitle={European Conference on Computer Vision},
  pages={56--72},
  year={2024},
  organization={Springer}
}
```
## Document
### Recommended Environment

- [x] torch 3.10.14
- [x] torchvision 0.17.2+cu121

```python
pip install pypi
pip install timm==1.0.7 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.8 albumentations==1.4.11 pytorch_wavelets==1.3.0 tidecv PyWavelets opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
```

### Train
You can choose BiFusionNet model in [BiFusionNet.yaml](../BiFusionNet/BiFusionNet.yaml)
Use the command python train.py to train the model





