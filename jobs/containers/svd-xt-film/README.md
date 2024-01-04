# svd-xt-film

## Build

```
pip install huggingface_hub[cli]
```

```
./dl-checkpoints.sh
cog build -t svd-xt-film
```

`--separate-weights` is omitted because we load weights from a Diffusers repo which includes other config files besides just model weights files.

## Run

```
docker run --network="host" --gpus all svd-xt-film
```