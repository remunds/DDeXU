# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 

RUN apt update && apt install -y python3-pip git wget curl vim

RUN pip3 install --upgrade pip

RUN pip3 install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install matplotlib scikit-learn tqdm
