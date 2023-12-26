# sdxl-turbo

## Build

```
./dl-checkpoints.sh
cog build --separate-weights -t sdxl-turbo
```

## Run

```
docker run --network="host" --gpus all sdxl-turbo
```