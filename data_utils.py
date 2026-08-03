import json
import os

import pandas as pd
import torch
from torchvision import datasets, transforms, models

DATASET_ROOTS = {"imagenet_val": "YOUR_PATH/ImageNet_val/",
                "broden": "data/broden1_224/images/"}

# Custom probing datasets. Anything registered here can be passed straight to
# --d_probe. For a one-off directory you do not want to register, use the
# `imagefolder:/path/to/dir` form instead (see get_data).
DATASET_ROOTS["treeoflife_train"] = "data/treeoflife/train"
DATASET_ROOTS["treeoflife_val"]   = "data/treeoflife/val"
DATASET_ROOTS["birds525_train"]   = "data/birds525/train"
DATASET_ROOTS["birds525_val"]     = "data/birds525/val"

# Datasets can also be registered without editing this file, by pointing
# CLIP_DISSECT_DATASETS at a JSON file of {"name": "/path/to/imagefolder"}.
# Useful on a cluster where the data lives at a path the repo shouldn't hardcode.
_DATASET_REGISTRY = os.environ.get("CLIP_DISSECT_DATASETS")
if _DATASET_REGISTRY and os.path.isfile(_DATASET_REGISTRY):
    with open(_DATASET_REGISTRY) as f:
        DATASET_ROOTS.update(json.load(f))


###############################################################################
# CLIP-side (probing) models
###############################################################################

# Short names for the open_clip models we care about. The value is whatever
# open_clip.create_model_and_transforms accepts as its first argument.
OPEN_CLIP_MODELS = {
    "bioclip":  "hf-hub:imageomics/bioclip",     # ViT-B/16, 512-d joint space
    "bioclip2": "hf-hub:imageomics/bioclip-2",   # ViT-L/14, 768-d joint space
}

# Architecture to instantiate when loading each model from a local checkpoint
# instead of the Hub. Both repos ship stock open_clip configs, so the named
# architecture is exactly the config the checkpoint was trained with.
OPEN_CLIP_ARCH = {
    "bioclip":  "ViT-B-16",
    "bioclip2": "ViT-L-14",
}

# Offline escape hatch: point these at a downloaded open_clip_pytorch_model.bin
# and nothing touches the network. LFCBM_BIOCLIP_CKPT is honoured too so a pod
# spec written for Label-free-CBM works here unchanged.
OPEN_CLIP_CKPT_ENV = {
    "bioclip":  ("CLIP_DISSECT_BIOCLIP_CKPT", "BIOCLIP_CKPT", "LFCBM_BIOCLIP_CKPT"),
    "bioclip2": ("CLIP_DISSECT_BIOCLIP2_CKPT", "BIOCLIP2_CKPT", "LFCBM_BIOCLIP2_CKPT"),
}


def _local_checkpoint(name):
    """Return the first existing checkpoint path named by this model's env vars."""
    for var in OPEN_CLIP_CKPT_ENV.get(name, ()):
        path = os.environ.get(var)
        if path and os.path.isfile(path):
            return path
    return None


def _resolve_open_clip(clip_name):
    """Map a --clip_model / --target_model string onto open_clip load arguments.

    Returns (model_id, pretrained) or None if the name is not an open_clip name.
    Accepted forms:
        bioclip | bioclip2            short names from OPEN_CLIP_MODELS
        hf-hub:org/repo               any open_clip model on the Hub
        open_clip:ARCH:PRETRAINED     e.g. open_clip:ViT-B-16:laion2b_s34b_b88k
                                      PRETRAINED may also be a local .bin/.pt path
    """
    if clip_name in OPEN_CLIP_MODELS:
        checkpoint = _local_checkpoint(clip_name)
        if checkpoint:
            return OPEN_CLIP_ARCH[clip_name], checkpoint
        return OPEN_CLIP_MODELS[clip_name], None

    if clip_name.startswith("hf-hub:"):
        return clip_name, None

    if clip_name.startswith("open_clip:"):
        parts = clip_name.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "open_clip names look like 'open_clip:ARCH:PRETRAINED', got {!r}".format(clip_name))
        return parts[1], parts[2]

    return None


def is_open_clip(clip_name):
    return _resolve_open_clip(clip_name) is not None


def load_open_clip(clip_name, device):
    """Load an open_clip model. Returns (model, preprocess, tokenizer)."""
    import open_clip

    model_id, pretrained = _resolve_open_clip(clip_name)
    model, _, preprocess = open_clip.create_model_and_transforms(model_id, pretrained=pretrained)

    # When a local checkpoint is loaded into a bare architecture, model_id is the
    # architecture name, so the tokenizer comes from there rather than the hub repo.
    tokenizer = open_clip.get_tokenizer(model_id)

    return model.to(device).eval(), preprocess, tokenizer


def get_clip_model(clip_name, device):
    """Return (model, preprocess, tokenizer) for the model that scores concepts.

    This is the model CLIP-Dissect uses to build the concept-activation matrix; it
    is independent of the target model being dissected. Any OpenAI CLIP name works
    as before ('ViT-B/16', 'RN50', ...); see _resolve_open_clip for the open_clip
    names, which is how you get BioCLIP and BioCLIP 2.

    The tokenizer is returned rather than assumed because open_clip models do not
    share OpenAI CLIP's global clip.tokenize.
    """
    if is_open_clip(clip_name):
        return load_open_clip(clip_name, device)

    import clip
    model, preprocess = clip.load(clip_name, device=device)
    return model.eval(), preprocess, clip.tokenize


###############################################################################
# Target models
###############################################################################

class CLIPVisionBackbone(torch.nn.Module):
    """The image tower of a CLIP-style model, exposed with dissectable layer names.

    Transformer blocks inside CLIP ViTs are awkward to hook directly: depending on
    the open_clip version their output is either (batch, tokens, dim) or the
    sequence-first (tokens, batch, dim), and utils.get_activation cannot tell the
    two apart. So each block is wrapped in a _BlockTap that normalises the layout
    and pools to (batch, dim) before handing the result to a named nn.Identity.
    Hooking those Identities means save_target_activations always sees a plain 2-D
    tensor and does no pooling of its own.

    Layer names available to --target_layers:
        block_0 ... block_{L-1}   output of each transformer block, pooled
        out                       pre-projection features (768-d B/16, 1024-d L/14)
        proj                      final joint image-text embedding (512-d / 768-d)
    """

    class _BlockTap(torch.nn.Module):
        def __init__(self, block, parent):
            super().__init__()
            self.block = block
            self.tap = torch.nn.Identity()
            # Plain attribute, not a submodule: registering the parent as a child
            # here would make the module graph cyclic.
            object.__setattr__(self, "_parent", parent)

        def forward(self, *args, **kwargs):
            out = self.block(*args, **kwargs)
            # Some open_clip versions return (x, attn_weights) from a block.
            tensor = out[0] if isinstance(out, tuple) else out
            self.tap(self._parent._pool_tokens(tensor))
            return out

    def __init__(self, clip_model, pool="cls"):
        super().__init__()
        self.visual = clip_model.visual
        self.pool = pool
        self.out = torch.nn.Identity()
        self.proj = torch.nn.Identity()
        self._batch = None
        self._ln_post_feat = {}

        # Input dtype: OpenAI CLIP keeps fp16 weights on cuda and casts inside
        # encode_image, which we bypass by calling visual() directly.
        self.dtype = next(self.visual.parameters()).dtype

        transformer = getattr(self.visual, "transformer", None)
        # Pre-2.24 open_clip always ran the transformer sequence-first; newer
        # versions declare the layout on the module itself.
        self.batch_first = bool(getattr(transformer, "batch_first", False))

        self.n_blocks = 0
        if transformer is not None and hasattr(transformer, "resblocks"):
            resblocks = transformer.resblocks
            for i in range(len(resblocks)):
                tap = self._BlockTap(resblocks[i], self)
                resblocks[i] = tap
                # Second reference to the same Identity so that `block_7` is a
                # valid attribute path for save_target_activations' eval(). The
                # Identity holds no parameters, so aliasing it costs nothing.
                setattr(self, "block_{}".format(i), tap.tap)
            self.n_blocks = len(resblocks)

        if hasattr(self.visual, "ln_post"):
            self.visual.ln_post.register_forward_hook(
                lambda module, inp, output: self._ln_post_feat.__setitem__("f", output))

    def _pool_tokens(self, x):
        if x.dim() != 3:
            return x.float()
        batch = self._batch
        # Prefer inferring the layout from the known batch size; fall back to the
        # declared one only when the token count and the batch size are equal.
        if x.shape[0] == batch and x.shape[1] != batch:
            seq = x
        elif x.shape[1] == batch and x.shape[0] != batch:
            seq = x.transpose(0, 1)
        else:
            seq = x if self.batch_first else x.transpose(0, 1)
        pooled = seq[:, 0] if self.pool == "cls" else seq.mean(dim=1)
        # .clone() is load-bearing. seq[:, 0] is a view into the block's full
        # (batch, tokens, dim) output, and .float() is a no-op when the model is
        # already fp32 -- so without an explicit copy every cached activation keeps
        # its entire parent tensor alive and the run OOMs after a few dozen batches.
        return pooled.float().clone()

    def layer_names(self):
        names = ["block_{}".format(i) for i in range(self.n_blocks)]
        if hasattr(self.visual, "ln_post"):
            names.append("out")
        names.append("proj")
        return names

    def forward(self, x):
        self._batch = x.shape[0]
        embedding = self.visual(x.to(self.dtype))
        if isinstance(embedding, tuple):
            embedding = embedding[0]

        if "f" in self._ln_post_feat:
            # ln_post runs after pooling in current open_clip (2-D) but over the
            # whole token sequence in older ones (3-D).
            self.out(self._pool_tokens(self._ln_post_feat.pop("f")))

        return self.proj(embedding.float())


#  timm backbones with the same ViT-B/16 architecture as BioCLIP but different
#  pretraining, so they isolate the pretraining objective from model scale.
TIMM_BACKBONES = {
    "vit_in21k":   "vit_base_patch16_224.augreg_in21k",  # supervised ImageNet-21k
    "dino_vitb16": "vit_base_patch16_224.dino",          # DINO self-supervised
}


class TimmBackbone(torch.nn.Module):
    """A timm vision model exposed as a feature extractor.

    `self.out` is an Identity whose output is the final pooled feature, so hooking
    it (--target_layers out) hands save_target_activations an already-2-D tensor.
    That keeps the pooling decision here instead of in utils.get_activation, which
    would otherwise mean-pool ViT tokens and silently discard the CLS token.
    """
    def __init__(self, model, pool="cls"):
        super().__init__()
        self.model = model
        self.pool = pool
        self.out = torch.nn.Identity()

    def forward(self, x):
        feats = self.model.forward_features(x)
        if feats.dim() == 3:                                  # (B, tokens, D)
            feats = feats[:, 0] if self.pool == "cls" else feats.mean(dim=1)
        elif feats.dim() == 4:                                # (B, D, H, W)
            feats = feats.mean(dim=[2, 3])
        # .clone() for the same reason as CLIPVisionBackbone._pool_tokens: the cls
        # slice is a view, and caching it would pin the whole token tensor.
        return self.out(feats.float().clone())


def load_timm_backbone(target_name, device, pool="cls"):
    """Load a timm backbone plus the preprocessing that model was trained with.

    num_classes=0 drops the classifier head. The transform comes from the model's
    own data config -- augreg_in21k and DINO use different normalisation, so a
    shared ImageNet transform would be wrong for one of them.
    """
    import timm
    model = timm.create_model(TIMM_BACKBONES[target_name], pretrained=True, num_classes=0)
    model = model.eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    preprocess = timm.data.create_transform(**cfg)
    return TimmBackbone(model, pool=pool).to(device).eval(), preprocess


def get_target_model(target_name, device):
    """
    returns target model in eval mode and its preprocess function

    target_name: supported options -
        bioclip, bioclip2         BioCLIP / BioCLIP 2 image towers (see CLIPVisionBackbone
                                  for the layer names these expose)
        hf-hub:org/repo           any open_clip model on the Hub, image tower only
        open_clip:ARCH:PRETRAINED any open_clip architecture + weights
        clip_ViT-B/16, clip_RN50  OpenAI CLIP image towers
        vit_in21k, dino_vitb16    timm ViT-B/16 baselines
        resnet18_places           ResNet-18 trained on Places-365
        resnet18/34/50/101/152,   models trained on ImageNet from torchvision
        vit_b_16, vit_b_32

    To dissect a different model implement its loading and preprocessing function here
    """
    if is_open_clip(target_name):
        model, preprocess, _ = load_open_clip(target_name, device)
        target_model = CLIPVisionBackbone(model).to(device).eval()

    elif target_name.startswith("clip_"):
        import clip
        model, preprocess = clip.load(target_name[5:], device=device)
        target_model = CLIPVisionBackbone(model.eval()).to(device).eval()

    elif target_name in TIMM_BACKBONES:
        target_model, preprocess = load_timm_backbone(target_name, device)

    elif target_name == 'resnet18_places':
        target_model = models.resnet18(num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    elif "vit_b" in target_name:
        target_name_cap = target_name.replace("vit_b", "ViT_B")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    elif "resnet" in target_name:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    else:
        raise ValueError("unknown target model {!r}; see get_target_model in data_utils.py "
                         "for the supported names".format(target_name))

    target_model.eval()
    return target_model, preprocess

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def get_data(dataset_name, preprocess=None):
    """Load a probing dataset.

    Besides the names registered in DATASET_ROOTS and the torchvision datasets
    below, `imagefolder:/path/to/dir` loads any directory laid out as
    class-subfolders-of-images without registering it first.
    """
    if dataset_name.startswith("imagefolder:"):
        root = dataset_name.split(":", 1)[1]
        if not os.path.isdir(root):
            raise FileNotFoundError("no such probing directory: {}".format(root))
        data = datasets.ImageFolder(root, preprocess)

    elif dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=preprocess)

    elif dataset_name == "cifar10_train":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=preprocess)

    elif dataset_name in ("places365_train", "places365_val"):
        split = 'train-standard' if dataset_name.endswith("train") else 'val'
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split=split, small=True,
                                      download=True, transform=preprocess)
        except RuntimeError:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split=split, small=True,
                                      download=False, transform=preprocess)

    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)

    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess),
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])

    else:
        raise ValueError(
            "unknown probing dataset {!r}. Registered names: {}. You can also pass "
            "imagefolder:/path/to/dir.".format(dataset_name, ", ".join(sorted(DATASET_ROOTS))))

    return data


def get_places_id_to_broden_label():
    with open("data/categories_places365.txt", "r") as f:
        places365_classes = f.read().split("\n")

    broden_scenes = pd.read_csv('data/broden1_224/c_scene.csv')
    id_to_broden_label = {}
    for i, cls in enumerate(places365_classes):
        name = cls[3:].split(' ')[0]
        name = name.replace('/', '-')

        found = (name+'-s' in broden_scenes['name'].values)

        if found:
            id_to_broden_label[i] = name.replace('-', '/')+'-s'
        if not found:
            id_to_broden_label[i] = None
    return id_to_broden_label

def get_cifar_superclass():
    cifar100_has_superclass = [i for i in range(7)]
    cifar100_has_superclass.extend([i for i in range(33, 69)])
    cifar100_has_superclass.append(70)
    cifar100_has_superclass.extend([i for i in range(72, 78)])
    cifar100_has_superclass.extend([101, 104, 110, 111, 113, 114])
    cifar100_has_superclass.extend([i for i in range(118, 126)])
    cifar100_has_superclass.extend([i for i in range(147, 151)])
    cifar100_has_superclass.extend([i for i in range(269, 281)])
    cifar100_has_superclass.extend([i for i in range(286, 298)])
    cifar100_has_superclass.extend([i for i in range(300, 308)])
    cifar100_has_superclass.extend([309, 314])
    cifar100_has_superclass.extend([i for i in range(321, 327)])
    cifar100_has_superclass.extend([i for i in range(330, 339)])
    cifar100_has_superclass.extend([345, 354, 355, 360, 361])
    cifar100_has_superclass.extend([i for i in range(385, 398)])
    cifar100_has_superclass.extend([409, 438, 440, 441, 455, 463, 466, 483, 487])
    cifar100_doesnt_have_superclass = [i for i in range(500) if (i not in cifar100_has_superclass)]

    return cifar100_has_superclass, cifar100_doesnt_have_superclass
