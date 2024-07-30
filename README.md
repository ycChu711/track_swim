Environment setup:

# Setup virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install torch torchvision opencv-python-headless pyyaml scipy matplotlib lapx filterpy seaborn gdown tqdm
pip install shapely

# Clone deep_sort_pytorch repository
git clone https://github.com/ZQPei/deep_sort_pytorch.git

gdown 1_qwTWdzT9dWNudpusgKavj_4elGgbkUN -O deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7

# Download pre-trained models
gdown 1gWuo2Vb8yOd4zW9XPKO3NLNcKkas7EnK -O best.torchscript_gpu.pt

gdown 1PXj2f-hUpe1NDU-AxZQaEcbixlHVaJOd -O best.torchscript_cpu.pt

gdown 1kpjazSleLIixFhsEMNVwt05i_yjl6uea -O coco.names
