#!/bin/bash
# EXPERIMENTAL STUFF, NOT NECESSARY! install CUDART 11.2 libraries
# sudo apt install cuda-cudart-11-2 cuda-cudart-dev-11-2
# if errors are encountered:
# sudo apt-get -o Dpkg::Options::="--force-overwrite" install --fix-broken

# Setup directories
cd ..
mkdir -p bts-workspace/dataset bts-workspace/
mv bts bts-workspace/code

# Setup environment
cd bts-workspace/code
python3 -m virtualenv venv-bts
source venv-bts/bin/activate
PYTHONPATH=""
python3 -m pip install --upgrade pip
python3 -m pip install tensorflow-gpu==1.13.1
python3 -m pip install -r tensorflow/requirements.txt
python3 -m pip install scikit-image
python3 -m pip install PySide2==5.12
python3 -m pip install pyopengl
python3 -m pip install pyglm


# Download model
mkdir -p tensorflow/models
python3 utils/download_from_gdrive.py 1nhukEgl3YdTBKVzcjxUp6ZFMsKKM3xfg tensorflow/models/bts_eigen_v2.zip
cd tensorflow/models && unzip bts_eigen_v2.zip

# Setup build
cd tensorflow/custom_layer/
mkdir build && cd build
# The code needs CUDA 10.0 (tested by the authors).
# To specify which cuda version you want to use with CMake, use 
# -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0
cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0 ..
make -j

