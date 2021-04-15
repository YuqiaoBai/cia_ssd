# DCOS config
remotely access dcos
```dcos task exec -it yunshuang1-1gpu bash```
```bash
export PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda-10.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-10.1/lib64:/usr/local/cuda-10.1/extras/CUPTI/lib64:
export CUDA_HOME=/usr/local/cuda-10.1
export LIBRARY_PATH=/usr/local/cuda-10.1/lib64/stubs
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

apt-get update -qq && apt-get install -y software-properties-common git nano

#! /bin/bash
apt-get install -y apt-utils cifs-utils  libboost-all-dev build-essential libssl-dev

# prepare data and project code
mkdir koko
mount -t cifs -o user=,password= //10.51.2.245/tmp/yunshuang koko
git clone https://yunshuang.yuan:$1@gitlab.uni-hannover.de/yunshuang.yuan/cia-ssd.git 

# build spconv
cd cia-ssd
pip install -r requirements.txt
cd spconv
python setup.py bdist_wheel
cd ./dist && pip install $(basename $./*.whl)

# build ops
cd ../.. && python setup.py develop

#login to wandb
wandb login

# run experiments
python run.py experiments cia_ssd_comap ../koko/data/synthdata3 ../koko/experiments-output/cia-ssd
```


