#!/bin/bash
ROOT_DIR=$(eval echo $1)
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
