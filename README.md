## CLIP-Dissect

An automatic and efficient tool to describe functionalities of individual neurons in DNNs.

This is the official repository for our paper: [CLIP-Dissect: Automatic Description of Neuron Representations in Deep Vision Networks](https://arxiv.org/abs/2204.10965) published at ICLR 2023. 

**Update 6/5/23**: We have conducted a crowdsourced evaluation of our description quality, results are available on [arxiv](https://arxiv.org/abs/2204.10965) (Appendix B).

![Overview](data/github_overview_figure.png)

## Installation

1. Install Python (3.10)
1. Install Pytorch (tested with 1.12.0, also works with 2.0) and Torchvision >= 0.13 following instructions from https://pytorch.org/get-started/previous-versions/
3. Install remaining requirements using `pip install -r requirements.txt`
4. Download the Broden dataset (images only) using `bash dlbroden.sh`
5. (Optional) Download ResNet-18 pretrained on Places-365: `bash dlzoo_example.sh`

We do not provide download instructions for ImageNet data, to evaluate using your own copy of ImageNet validation set you must set 
the correct path in `DATASET_ROOTS["imagenet_val"]` variable in `data_utils.py`.

## Quickstart:

This will dissect 5 layers of ResNet-50(ImageNet) using Broden as the probing dataset. Results will be saved in `results/resnet50_{datetime}/descriptions.csv`.

```
python describe_neurons.py
```

## Recreating experiments

The results used for figures and tables of our paper can be recreated by running the corresponding notebook in the `experiments` folder, for example to reproduce Table 1 run `experiments/table1.ipynb`.

## BioCLIP and BioCLIP 2

This fork adds [BioCLIP](https://huggingface.co/imageomics/bioclip) and
[BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) — and any other
[open_clip](https://github.com/mlfoundations/open_clip) checkpoint — in both of the
roles CLIP-Dissect has. Those roles are independent and it is worth keeping them
straight:

- **`--clip_model`** is the model that *scores concepts*. It embeds the probing
  images and the concept strings into a joint space to build the concept-activation
  matrix.
- **`--target_model`** is the model whose *neurons are being described*.

If your concept set is biological, `--clip_model` matters most. OpenAI CLIP's text
encoder has seen very few taxonomic names, so scoring a concept set of species
against it mostly produces noise; BioCLIP was trained on exactly those captions.

### Model names

Both flags accept the same set of names:

| Name | Model |
| --- | --- |
| `bioclip` | BioCLIP, ViT-B/16, 512-d joint space |
| `bioclip2` | BioCLIP 2, ViT-L/14, 768-d joint space |
| `hf-hub:org/repo` | any open_clip model on the HuggingFace Hub |
| `open_clip:ARCH:PRETRAINED` | any open_clip arch + weights, e.g. `open_clip:ViT-B-16:laion2b_s34b_b88k`. `PRETRAINED` may be a local `.bin`/`.pt` path |
| `ViT-B/16`, `RN50`, ... | OpenAI CLIP, as before |

`--target_model` additionally takes `clip_ViT-B/16` (OpenAI CLIP's image tower),
the timm baselines `vit_in21k` and `dino_vitb16`, and the original torchvision /
`resnet18_places` options.

### Layer names for CLIP-style targets

Transformer blocks emit token sequences whose layout differs between open_clip
versions, so the image tower is wrapped in `data_utils.CLIPVisionBackbone`, which
exposes pooled, batch-first taps under stable names:

- `block_0` … `block_{L-1}` — output of each transformer block (12 for BioCLIP, 24 for BioCLIP 2)
- `out` — pre-projection features (768-d for BioCLIP, 1024-d for BioCLIP 2)
- `proj` — the final joint image-text embedding (512-d / 768-d)

To see them for any target model, along with the actual feature width of each:

```
python list_layers.py --target_model bioclip2
```

### Examples

Dissect BioCLIP's last blocks with BioCLIP itself scoring the concepts:

```
python describe_neurons.py --clip_model bioclip --target_model bioclip \
    --target_layers block_9,block_10,block_11,out \
    --d_probe imagenet_val --concept_set data/20k.txt
```

Dissect a ResNet-50 on a generic dataset, unchanged from upstream:

```
python describe_neurons.py --target_model resnet50 --d_probe broden --concept_set data/20k.txt
```

Compare what BioCLIP 2 and OpenAI CLIP say about the same neurons — the activations
are cached per probing model, so the second run reuses the target activations:

```
python describe_neurons.py --clip_model bioclip2 --target_model resnet50 --target_layers layer4
python describe_neurons.py --clip_model ViT-B/16 --target_model resnet50 --target_layers layer4
```

Results go to `results/{target_model}_{clip_model}_{datetime}/descriptions.csv`.

### Running offline / from a local checkpoint

By default the BioCLIP weights are pulled from the Hub. To use a checkpoint you
already have, set one of these to a downloaded `open_clip_pytorch_model.bin` and
nothing touches the network:

```
export CLIP_DISSECT_BIOCLIP_CKPT=/workspace/models/bioclip/open_clip_pytorch_model.bin
export CLIP_DISSECT_BIOCLIP2_CKPT=/workspace/models/bioclip-2/open_clip_pytorch_model.bin
```

`BIOCLIP_CKPT` / `BIOCLIP2_CKPT` and `LFCBM_BIOCLIP_CKPT` / `LFCBM_BIOCLIP2_CKPT` are
also honoured, so an environment set up for Label-free-CBM works here unchanged. Both
repos ship stock open_clip configs, so the checkpoint is loaded into a plain
`ViT-B-16` / `ViT-L-14` architecture.

## How to modify:

### Dissecting your own model

1. Implement the code to load your model(in eval mode) and a preprocess function to correctly load images for your model in `get_target_model` function of `data_utils.py` under an if statement for target_name of you choice. 
2. Dissect the model by running `python describe_neurons.py --target_model {model_name}`

### Using your own probing dataset

`--d_probe` takes any of the names registered in `data_utils.DATASET_ROOTS`
(`imagenet_val`, `broden`, `imagenet_broden`, `cifar10_train/val`,
`cifar100_train/val`, `places365_train/val`, ...). For your own data there are three
options, in increasing order of permanence:

1. **Ad hoc** — point at a directory of class subfolders, no code change:
   ```
   python describe_neurons.py --d_probe imagefolder:/path/to/my_dataset
   ```
2. **Per-environment** — register paths in a JSON file and export
   `CLIP_DISSECT_DATASETS=/path/to/datasets.json`:
   ```json
   {"my_dataset_train": "/workspace/data/my_dataset/train"}
   ```
   This keeps cluster-specific paths out of the repo.
3. **Permanent** — add an entry to `DATASET_ROOTS` in `data_utils.py`, or, if the
   dataset is not a plain ImageFolder, implement its loading in `get_data`.

Anything not laid out as class subfolders needs `get_data` extended by hand.

Note that the default `--similarity_fn soft_wpmi` ranks the 100 most-activating
probe images per neuron, so it needs a probing set of at least 100 images (`wpmi`
needs 28). Smaller sets should use `--similarity_fn cos_similarity`.

### Using your own concept set

1. Create/download a .txt file containing your concept set, with each concept on a separate line. Blank lines and surrounding whitespace are ignored.
2. Dissect the model by running `python describe_neurons.py --concept_set {path_to_conceptset}`

Concepts are embedded as bare strings by default. `--prompt_template` wraps each one
first, which can matter for models trained on full captions:

```
python describe_neurons.py --clip_model bioclip --concept_set data/my_species.txt \
    --prompt_template "a photo of {}."
```

### Specifying device

You can specify which device is used with the `--device` argument, which defaults to `cuda`, i.e. `python describe_neurons.py --device cpu`

### DataLoader workers

Activation extraction uses 8 DataLoader workers, which needs a `/dev/shm` larger
than the 64Mi a container gets by default and is wasted on a low-CPU pod. Override
with `CLIP_DISSECT_NUM_WORKERS=2` (or `0` to load in the main process).

## Sources:

- CLIP: https://github.com/openai/CLIP
- Text datasets(10k and 20k): https://github.com/first20hours/google-10000-english
- Text dataset(3k): https://www.ef.edu/english-resources/english-vocabulary/top-3000-words/
- Broden download script based on: https://github.com/CSAILVision/NetDissect-Lite

## Common errors

**Incorrect activations cached:**

The code automatically caches the saved activations of target model and CLIP in `saved_activations`, and if a file already exists with the same save name the code will load these activations instead of recalculating. However sometimes you may wish to modify the pipeline in a way that doesn't change the name of the saved activations and want to recalculate the activations. In this case you need to manually delete the relevant files from saved_activations before rerunning CLIP-Dissect, as using incorrect activations will give incorrect results.

## Cite this work

T. Oikarinen and T.-W. Weng, CLIP-Dissect: Automatic Description of Neuron Representations in Deep Vision Networks, ICLR 2023.

```
@article{oikarinen2023clip,
  title={CLIP-Dissect: Automatic Description of Neuron Representations in Deep Vision Networks},
  author={Oikarinen, Tuomas and Weng, Tsui-Wei},
  journal={International Conference on Learning Representations},
  year={2023}
}
```
