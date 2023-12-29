# sdxl-turbo

## Build

```
./dl-checkpoints.sh
cog build -t sdxl-turbo
```

`--separate-weights` is omitted because we load weights from a Diffusers repo which includes other config files besides just model weights files.

## Run

```
docker run --network="host" --gpus all sdxl-turbo
```