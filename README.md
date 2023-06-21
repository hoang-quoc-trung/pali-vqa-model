# PaLI model
PaLI is a unified language-image model trained to perform many tasks and in over 100 languages.
Paper: [PaLI: A Jointly-Scaled Multilingual Language-Image Model (2022)](https://arxiv.org/pdf/2209.06794.pdf)

![PaLI Architecture](./docs/PaLI.gif)

## Configuration
We use `.yaml` file to save to configs of dataset and model architecture, take a look at the `src/configs/default_config.yml` to get an example:
```yaml
---
---
dataset:
  training:
    data_root: '...'
    csv_file: '...'
  validation:
    data_root: '...'
    csv_file: '...'
  testing:
    data_root: '...'
    csv_file: '...'
hyperparams:
  image_size: [224, 224]
  encoder_input_tokens: 38
  decoder_input_tokens: 18
  decoder_target_tokens: 18
  learning_rate: 0.0007
  grad_norm_clip: 1.0
  train_batch_size: 64
  val_batch_size: 64
  num_steps: 500000
  save_steps: 5000
  val_steps: 2000
  save_checkpoint_dir: './src/checkpoints/state'
  load_checkpoint_dir: '...'
  load_params_dir: '...'
  save_params_dir: './src/checkpoints/params'
  save_history_dir: './src/checkpoints/history_log'
t5:
  gin_file: 'models/t5x/t5x/examples/t5/t5_1_1/base.gin'
  dropout_rate: 0.0
  pretrained_path: null
  vit_patches_embs_size: 50
  pretrained_path: '../pretrained/flan_t5_base'
  vocab_size: 32128
vit:
  num_classes: null
  model_name: 'ViT-B_32'
  classifier: "token_unpooled"
  pretrained_path: '../pretrained/imagenet21k_ViT-B_32.npz'
```

There are a few points you need to pay attention to:

1. The `emb_size` in T5 model and `hidden_size` in ViT model must be the same cause the idea behind PaLI is **concatenating the dense token embeddings with the patch embeddings**. 
2. Two parameters: `num_classes` and `classifier` is important if you want to set the kind of the output of ViT model. If you want your output is:

    - *Classification*: 
        - `num_classes: <number_of_classes>`
        - `classifier: 'token'`
    - *[CLS] token embedding*:
        - `num_classes: null`
        - `classifier: 'unpooled'`.
    - *Patches embeddings*: 
        - `num_classes: null`
        - `classifier: 'token_unpooled'`.
3. Because of the embedding shape of them, you also need to choose **a suitable pretrained models**.

## Download Pretrained Models
- ViT: [GCS](https://console.cloud.google.com/storage/browser/vit_models/imagenet21k?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)
- T5: [GCS](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)

###### Examples:
  - ViT base 32
  ```bash
  cd ./pretrained
  wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz
  ```
  - Flan T5 base
  ```bash
  cd ./pretrained
  gsutil -m cp -r \
  "gs://t5-data/pretrained_models/t5x/flan_t5_base" \
  .
  ```
## Installation (linux)
- Install cuda 11.8:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
sudo rm /bin/nvidia-smi
sudo ln -s /opt/bin/nvidia-smi /bin/
```

- Install environments:
```bash
mamba env create -f environment.yml
mamba activate VisualQA
pip install git+https://github.com/google/flaxformer

export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/include:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
```

## Dataset Pattern
Currently, we just create dataset pattern for task Visual QA only, please follow this pattern:

- Save all images in a folder, replace the path as `data_root` in config file.
- Prepare a `CSV file` with 3 columns:
    - `image`: the name of image file.
    - `question`
    - `answer`

Samples:
|question|answer|image|
|:--:|:--:|:--:|
|Where is he looking|down|COCO_val2014_000000262148.jpg
What are the people in the background doing?|watching|COCO_val2014_000000262148.jpg
What is he on top of?|picnic table|COCO_val2014_000000262148.jpg
What website copyrighted the picture?|foodiebakercom|COCO_val2014_000000393225.jpg
|Is this a creamy soup?|no|COCO_val2014_000000393225.jpg


## Training PaLI model
```
usage: train.py [-h] [--config_path CONFIG_PATH] [--set_memory_growth] [--gpu_memory_fraction GPU_MEMORY_FRACTION] [--gpu_id GPU_ID]
                      [--seed SEED]

PaLI training script

optional arguments:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path to the config file
  --set_memory_growth
  --gpu_memory_fraction GPU_MEMORY_FRACTION
  --gpu_id GPU_ID       GPU device id
  --seed SEED           Random seed
```

## Training PaLI model - Multi GPU
```
usage: train_multi_gpu.py [-h] [--config_path CONFIG_PATH] [--set_memory_growth] [--gpu_memory_fraction GPU_MEMORY_FRACTION] [--gpu_id GPU_ID]
                      [--seed SEED]

PaLI training script

optional arguments:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path to the config file
  --set_memory_growth
  --gpu_memory_fraction GPU_MEMORY_FRACTION
  --gpu_id GPU_ID       GPU device id
  --seed SEED           Random seed
```