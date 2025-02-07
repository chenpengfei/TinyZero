# -2 install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc


# -1 install wandb
pip install wandb
wandb login
# TinyZero-Qwen2.5-7B: 19fffff6dedf30d8f82534a9cb9f55f639a1839f

# 0 download base model
mkdir -p /lpai/models
apt-get update
apt-get install git-lfs
git clone https://huggingface.co/Qwen/Qwen2.5-7B /lpai/models/Qwen2.5-7B

# 1 Installation
conda create -n zero

# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==24.12
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib

# 2 Countdown task

# Data Preparation
conda activate zero
mkdir -p /lpai/dataset/countdown
python ./examples/data_preprocess/countdown.py --local_dir /lpai/datasets/countdown

# 3 Run Training
conda activate zero

# 3B+ model, In this case, the base model is able to develop sophisticated reasoning skills.
export N_GPUS=2
export BASE_MODEL=/lpai/models/Qwen2.5-7B
export DATA_DIR=/lpai/datasets/countdown
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-7b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
