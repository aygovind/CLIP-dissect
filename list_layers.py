"""Print the layer names a target model accepts in --target_layers.

CLIP-Dissect resolves a layer name by attribute path (`target_model.<name>`), so
the legal names are model-specific and not otherwise discoverable. For CLIP-style
image towers, prefer the curated list this prints over the raw module tree: the
raw transformer blocks emit token sequences whose layout varies by open_clip
version, and CLIPVisionBackbone wraps each one in a pooled, batch-first tap.

Usage:
    python list_layers.py --target_model bioclip
    python list_layers.py --target_model resnet50 --all
"""
import argparse

import torch

import data_utils

parser = argparse.ArgumentParser(description="List dissectable layer names for a target model")
parser.add_argument("--target_model", type=str, required=True,
                    help="Same values as describe_neurons.py --target_model")
parser.add_argument("--device", type=str, default="cpu",
                    help="Device to instantiate the model on; cpu is enough to read names")
parser.add_argument("--all", action="store_true",
                    help="Also dump every named submodule, not just the recommended taps")


def main():
    args = parser.parse_args()
    model, preprocess = data_utils.get_target_model(args.target_model, args.device)

    print("target_model: {}".format(args.target_model))
    print("preprocess:   {}".format(str(preprocess).replace("\n", " ")))
    print()

    if hasattr(model, "layer_names"):
        names = model.layer_names()
        print("recommended --target_layers (pooled, batch-first taps):")
        for name in names:
            print("    {}".format(name))
        print()
        print("e.g. --target_layers {}".format(",".join(names[-4:])))
    else:
        top_level = [name for name, _ in model.named_children()]
        print("top-level modules (usable directly in --target_layers):")
        for name in top_level:
            print("    {}".format(name))

    if args.all:
        print()
        print("all named submodules:")
        for name, module in model.named_modules():
            if name:
                print("    {:<50} {}".format(name, module.__class__.__name__))

    # Feature widths make it obvious whether a layer is the one you meant, and
    # catch a preprocess/architecture mismatch before a full dissection run.
    print()
    print("output widths on a dummy batch:")
    widths = {}
    hooks = []
    named = model.layer_names() if hasattr(model, "layer_names") else \
        [name for name, _ in model.named_children()]
    for name in named:
        module = model
        for part in name.split("."):
            module = getattr(module, part)
        hooks.append(module.register_forward_hook(
            lambda m, i, o, n=name: widths.__setitem__(n, tuple(o.shape[1:]))))
    with torch.no_grad():
        model(torch.zeros(2, 3, 224, 224, device=args.device))
    for hook in hooks:
        hook.remove()
    for name in named:
        print("    {:<20} {}".format(name, widths.get(name, "(not reached)")))


if __name__ == "__main__":
    main()
