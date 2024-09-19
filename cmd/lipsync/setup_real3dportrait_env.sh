#!/bin/bash

# Check if the root directory is passed as an argument
if [ -z "$1" ]; then
  echo "Please provide the root directory as an argument."
  exit 1
fi

ROOT_DIR=$(eval echo $1)

# Navigate to the repository directory
cd ${ROOT_DIR}/models/
mkdir models--yerfor--Real3DPortrait
cd models--yerfor--Real3DPortrait

# Required for downloading real3dportrait checkpoints
pip install gdown

# Create the required directories
mkdir -p deep_3drecon/BFM
mkdir -p checkpoints/240126_real3dportrait_orig/audio2secc_vae
mkdir -p checkpoints/240126_real3dportrait_orig/secc2plane_torso_orig
mkdir -p checkpoints/pretrained_ckpts

echo "Directory structure created."

# Download 3DMM BFM Model
echo "Downloading 3DMM BFM Model..."
cd deep_3drecon/BFM
gdown https://drive.google.com/uc?id=1SPM3IHsyNAaVMwqZZGV6QVaV7I2Hly0v
gdown https://drive.google.com/uc?id=1MSldX9UChKEb3AXLVTPzZQcsbGD4VmGF
gdown https://drive.google.com/uc?id=180ciTvm16peWrcpl4DOekT9eUQ-lJfMU
gdown https://drive.google.com/uc?id=1KX9MyGueFB3M-X0Ss152x_johyTXHTfU
gdown https://drive.google.com/uc?id=19-NyZn_I0_mkF-F5GPyFMwQJ_-WecZIL
gdown https://drive.google.com/uc?id=11ouQ7Wr2I-JKStp2Fd1afedmWeuifhof
gdown https://drive.google.com/uc?id=18ICIvQoKX-7feYWP61RbpppzDuYTptCq
gdown https://drive.google.com/uc?id=1VktuY46m0v_n_d4nvOupauJkK4LF6mHE
cd ../..

# Download Pre-trained Real3D-Portrait models
echo "Downloading Pre-trained Real3D-Portrait models..."
cd checkpoints
gdown https://drive.google.com/uc?id=1gz8A6xestHp__GbZT5qozb43YaybRJhZ
gdown https://drive.google.com/uc?id=1gSUIw2AkkKnlLJnNfS2FCqtaVw9tw3QF
unzip 240210_real3dportrait_orig.zip
unzip pretrained_ckpts.zip
ls
cd ..
# Navigate to the repository directory
cd ${ROOT_DIR}/models/models--yerfor--Real3DPortrait
# Check if all required files are in place
required_files=(
    "deep_3drecon/BFM/01_MorphableModel.mat"
    "deep_3drecon/BFM/BFM_exp_idx.mat"
    "deep_3drecon/BFM/BFM_front_idx.mat"
    "deep_3drecon/BFM/BFM_model_front.mat"
    "deep_3drecon/BFM/Exp_Pca.bin"
    "deep_3drecon/BFM/facemodel_info.mat"
    "deep_3drecon/BFM/std_exp.txt"
    "deep_3drecon/reconstructor_opt.pkl"
    "checkpoints/240210_real3dportrait_orig/audio2secc_vae/config.yaml"
    "checkpoints/240210_real3dportrait_orig/audio2secc_vae/model_ckpt_steps_400000.ckpt"
    "checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig/config.yaml"
    "checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig/model_ckpt_steps_100000.ckpt"
    "checkpoints/pretrained_ckpts/mit_b0.pth"
)

all_files_exist=true

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "File not found: $file"
        all_files_exist=false
    fi
done

if $all_files_exist; then
    echo "All files are in the correct places."
else
    echo "Some files are missing."
fi

cd ..
mv $CONDA_DIR $REPO_DIR/anaconda3
