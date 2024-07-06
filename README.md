# ConvNeXt_Install
# Installation

We provide installation instructions for ImageNet classification experiments here.

## Dependency Setup
Create an new conda virtual environment
```
conda create -n convnext python=3.8 -y
conda activate convnext
```

Install CUDA
```
conda install cudatoolkit=11.1.1 -c conda-forge
```
Check cuDNN installed. If not, install cuDNN (older version than CUDA) <br>
![image](https://github.com/noyaboy/ConvNeXtV2_Install/assets/99811508/2760601b-d92a-45f3-b1cd-341f84e685d2)
```
conda list
conda search cudnn -c conda-forge
conda install cudnn==8.1.0.77 -c conda-forge
```
Check CUDA installed. If not, install cudatoolkit-dev. Related to $CUDA_HOME variable issue)
```
nvcc --version
conda search cudatoolkit-dev -c conda-forge
conda install cudatoolkit-dev=11.1.1 -c conda-forge
```
Install [Pytorch](https://pytorch.org/)>=1.8.0, [torchvision](https://pytorch.org/vision/stable/index.html)>=0.9.0 following official instructions. <br>
Go to https://pytorch.org/get-started/previous-versions/ and search for the required PyTorch version
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
Go to https://blog.csdn.net/qq_38825788/article/details/125859041 to authorize the GitHub account.
```
ssh -T git@github.com
```
>> git@github.com: Permission denied (publickey) <br>

Way to authorize the GitHub account
```
ssh-keygen -t rsa
```
Three enters <br>
then to ~/.ssh/id_rsa.pub copy content, login github, press setting, SSH and GPG keys, New SSH keys, random title, paste content to key, press Add key
```
ssh -T git@github.com
```
>> Hi <username>, You've successfully authenticated, but GitHub does not provide shell access.


Clone this repo and install required packages:
```
git clone https://github.com/facebookresearch/ConvNeXt
pip install timm==0.3.2 tensorboardX six
```
Add Makefile
```
cd ~/ConvNeXt
touch Makefile
vim Makefile
```
```
train:
	python -m torch.distributed.launch --nproc_per_node=8 main.py \
	--model convnext_tiny --drop_path 0.1 \
	--batch_size 64 --lr 4e-3 --update_freq 4 \
	--model_ema true --model_ema_eval true \
	--data_path /home1/dataset/ImageNet/ \
	--output_dir /home1/science103555/ckp_weight/ConvNeXt/
```
train
```
make train
```
Export environment
```
cd ~
conda env export > convnext.yml
```

