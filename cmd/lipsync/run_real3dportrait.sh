#!/bin/bash

activate_conda_env() {
  local env_path="$1"
  if [ -f "${env_path}/etc/profile.d/conda.sh" ]; then
    source "${env_path}/etc/profile.d/conda.sh"
    conda activate "${env_path}"
  else
    echo "Conda activation script not found at ${env_path}/etc/profile.d/conda.sh"
    exit 1
  fi
}

activate_conda_env "/models/models--yerfor--Real3DPortrait/anaconda3"

# Run the Real3DPortrait inference command
PYTHONPATH=/models/models--yerfor--Real3DPortrait/anaconda3/bin/python /models/models--yerfor--Real3DPortrait/inference/real3d_infer.py \
  --src_img "$1" \
  --drv_aud "$2" \
  --out_name "$3" \
  --out_mode final
