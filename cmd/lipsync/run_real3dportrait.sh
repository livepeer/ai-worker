#!/bin/bash
activate_conda_env() {
  local env_path="$1"
  if [ -f "${env_path}/etc/profile.d/conda.sh" ]; then
    source "${env_path}/etc/profile.d/conda.sh"
    source "${env_path}/bin/conda init bash"
    conda activate real3dportrait
  else
    echo "Conda activation script not found at ${env_path}/etc/profile.d/conda.sh"
    exit 1
  fi
}

activate_conda_env "/models/models--yerfor--Real3DPortrait/anaconda3"

# Debugging: Check which Python is being used - 3.9 is used by the conda env for real3dportrait
which python
python --version

# Set PYTHONPATH
export PYTHONPATH="/models/models--yerfor--Real3DPortrait:$PYTHONPATH"
cd /models/models--yerfor--Real3DPortrait

# Run the Real3DPortrait inference command
python /models/models--yerfor--Real3DPortrait/inference/real3d_infer.py \
  --src_img "$1" \
  --drv_aud "$2" \
  --out_name "$3" \
  --drv_pose "$4" \
  --out_mode final

