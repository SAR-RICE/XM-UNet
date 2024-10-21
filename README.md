<div align="center">
<h1>eXplainable Mamba UNet - XM-UNet </h1>
<h3>The official repository of XM-UNet. We will update this repository with the publication details and release the code as soon as the related paper is published.</h3>

</div>


## 🔥🔥Highlights🔥🔥
### *1.The XM-UNet has only 0.019M parameters, 0.279 GFLOPs.*</br>
### *2.The first explainable model for SAR rice area extraction.*</br>

**0. Main Environments.** </br>
The environment installation procedure can be followed by the steps below (python=3.8):</br>
```
conda create -n xmunet python=3.8
conda activate xmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```
