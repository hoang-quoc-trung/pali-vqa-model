# PaLI model
PaLI is a unified language-image model trained to perform many tasks and in over 100 languages.

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
  batch_size: 2
  learning_rate: 0.0001
  num_steps: 50000
  save_steps: 5000
  val_steps: 2000
  save_checkpoint_dir: './src/checkpoints'
  load_checkpoint_dir: './src/checkpoints'
t5:
  gin_file: 'models/t5x/t5x/examples/t5/t5_1_1/base.gin'
  dropout_rate: 0.0
  pretrained_path: null
  vit_patches_embs_size: 50
  pretrained_path: '../pretrained/t5_1_1_base'
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

## Pretrained models
- ViT: [GCS](https://console.cloud.google.com/storage/browser/vit_models/imagenet21k?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)
- T5: [GCS](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)

## Installation
Install cuda 11.8 and cudnn:
```bash
mamba create -n VisualQA python==3.8.* -y
mamba install cudatoolkit==11.8 cudnn==8.4.1.50 -y
```

Install jax CPU or GPU version:
```bash
python setup.py install
```

## Dataset pattern
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
usage: train_model.py [-h] [--config_path CONFIG_PATH] [--set_memory_growth] [--gpu_memory_fraction GPU_MEMORY_FRACTION] [--gpu_id GPU_ID]
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
