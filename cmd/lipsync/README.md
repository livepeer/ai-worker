# Setup for lipsync pipeline

Real3DPortrait https://github.com/yerfor/Real3DPortrait is utilized for the lipsync pipeline

Conda environment is built and configured on the host at /models/models--yerfor--Real3DPortrait/anaconda3 to mimic the paths that will be encountered by the containerized app.

Usage:


``` sudo ./setup_real3dportrait_env.sh ~```

Models are expected to be stored at ~/.lpData/models/ on the host machine.