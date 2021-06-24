#!/bin/bash
BTS_PARENT_DIR=$(dirname "${PWD}")
BTS_CODE_DIR="${PWD}"
WORKSPACE_DIR="${BTS_PARENT_DIR}/bts_workspace"
NEW_BTS_CODE_DIR="${WORKSPACE_DIR}/code"

# Setup directories
cd ${BTS_PARENT_DIR}
mkdir -p "${WORKSPACE_DIR}/dataset"
mv "${BTS_CODE_DIR}" "${NEW_BTS_CODE_DIR}"
BTS_CODE_DIR="${NEW_BTS_CODE_DIR}"
unset NEW_BTS_CODE_DIR

# Setup virtual environment and dependencies
cd "${BTS_CODE_DIR}"
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv

python3 -m virtualenv venv-bts
source "${BTS_CODE_DIR}/venv-bts/bin/activate"
PYTHONPATH=""
python3 -m pip install -r "${BTS_CODE_DIR}/tensorflow/requirements.txt"

# Download model from Google Drive
TF_MODELS_DIR="${BTS_CODE_DIR}/tensorflow/models"
mkdir -p "${TF_MODELS_DIR}"
echo "Downloading Tensorflow KITTI pre-trained model..."
python3 "${BTS_CODE_DIR}/utils/download_from_gdrive.py" \
  1nhukEgl3YdTBKVzcjxUp6ZFMsKKM3xfg "${TF_MODELS_DIR}/bts_eigen_v2.zip"
cd "${TF_MODELS_DIR}" && unzip bts_eigen_v2.zip

# Setup build
TF_CUSTOM_LAYER_DIR="${BTS_CODE_DIR}/tensorflow/custom_layer"
cd "${TF_CUSTOM_LAYER_DIR}"
BTS_BUILD_DIR="${TF_CUSTOM_LAYER_DIR}/build"
echo "Setting up build directories and options..."
mkdir -p "${BTS_BUILD_DIR}" && cd "${BTS_BUILD_DIR}"
# The code needs CUDA 10.0 (tested by the authors).
# To specify which cuda version you want to use with CMake, use 
# -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0
cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0 ..
make -j

cd "${BTS_CODE_DIR}"
