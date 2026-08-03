"""Compare what one model's neurons look like across several probing datasets.

describe_neurons.py answers "what does neuron N detect?" for a single D_probe.
This answers "does that answer change when I swap D_probe?", which is the question
when you want to know whether a description is a property of the neuron or an
artefact of the images you probed it with.

For each requested neuron it writes one figure: one row per probing dataset,
showing that dataset's top-activating images alongside the concept it produced.

Activations are built if missing and reused if present, so running this after
describe_neurons.py on the same settings recomputes nothing.

Usage:
    python compare_probes.py --target_model bioclip2 --clip_model bioclip2 \
        --target_layer block_23 --concept_set data/20k.txt \
        --d_probes broden,treeoflife_val,birds525_val --n_neurons 12
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")            # write files without needing a display
from matplotlib import pyplot as plt
import torch

import data_utils
import similarity as similarity_module
import utils

parser = argparse.ArgumentParser(description="Compare neuron descriptions across probing datasets")
parser.add_argument("--clip_model", type=str, default="bioclip2")
parser.add_argument("--target_model", type=str, default="bioclip2")
parser.add_argument("--target_layer", type=str, default="block_23",
                    help="A single layer; run once per layer you care about")
parser.add_argument("--d_probes", type=str, required=True,
                    help="Comma-separated probing datasets to compare (no spaces)")
parser.add_argument("--concept_set", type=str, default="data/20k.txt")
parser.add_argument("--similarity_fn", type=str, default="soft_wpmi",
                    choices=["soft_wpmi", "wpmi", "rank_reorder", "cos_similarity",
                             "cos_similarity_cubed"])
parser.add_argument("--n_neurons", type=int, default=10,
                    help="How many neurons to plot when --neurons is not given")
parser.add_argument("--neurons", type=str, default=None,
                    help="Comma-separated neuron indices; overrides --n_neurons")
parser.add_argument("--select", type=str, default="random", choices=["random", "top", "disagree"],
                    help="""How to pick neurons: 'random' samples them, 'top' takes the
                         highest-similarity ones on the first probing set, 'disagree' takes
                         the ones whose predicted concept varies most across probing sets""")
parser.add_argument("--n_images", type=int, default=5, help="Top activating images per row")
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--pool_mode", type=str, default="avg")
parser.add_argument("--activation_dir", type=str, default="saved_activations")
parser.add_argument("--out_dir", type=str, default="results/comparisons")
parser.add_argument("--seed", type=int, default=0)


def label_for(d_probe):
    """Short display name for a probing set.

    A registered name is already short; an `imagefolder:` path is not, and printing
    it in full swamps the figure titles and the csv headers. Keep the last two path
    components so .../treeoflife/val and .../birds525/val stay distinguishable.
    """
    if not d_probe.startswith("imagefolder:"):
        return d_probe
    parts = [p for p in d_probe.split(":", 1)[1].split("/") if p]
    return "/".join(parts[-2:]) if parts else d_probe


def collect(args, d_probe, words):
    """Build (or reuse) activations for one probing set and return what we plot."""
    utils.save_activations(clip_name=args.clip_model, target_name=args.target_model,
                           target_layers=[args.target_layer], d_probe=d_probe,
                           concept_set=args.concept_set, batch_size=args.batch_size,
                           device=args.device, pool_mode=args.pool_mode,
                           save_dir=args.activation_dir)

    save_names = utils.get_save_names(clip_name=args.clip_model, target_name=args.target_model,
                                      target_layer=args.target_layer, d_probe=d_probe,
                                      concept_set=args.concept_set, pool_mode=args.pool_mode,
                                      save_dir=args.activation_dir)

    similarity_fn = getattr(similarity_module, args.similarity_fn)
    similarities, target_feats = utils.get_similarity_from_activations(
        *save_names, similarity_fn, return_target_feats=True, device=args.device)

    vals, ids = torch.max(similarities, dim=1)
    n_images = min(args.n_images, target_feats.shape[0])
    _, top_ids = torch.topk(target_feats, k=n_images, dim=0)

    return {
        "name": d_probe,
        "label": label_for(d_probe),
        # No transform, so the dataset yields PIL images we can draw directly.
        "data": data_utils.get_data(d_probe),
        "concepts": [words[int(i)] for i in ids],
        "scores": vals.cpu(),
        "top_ids": top_ids.cpu(),
        "n_neurons": target_feats.shape[1],
    }


def pick_neurons(args, probes):
    if args.neurons:
        return [int(n) for n in args.neurons.split(",")]

    n_neurons = min(p["n_neurons"] for p in probes)
    if args.select == "top":
        order = torch.argsort(probes[0]["scores"][:n_neurons], descending=True)
        return order[:args.n_neurons].tolist()

    if args.select == "disagree":
        # Neurons where the probing sets disagree most are the interesting ones: a
        # description that survives every D_probe is not what you are looking for here.
        spread = []
        for n in range(n_neurons):
            distinct = {p["concepts"][n] for p in probes}
            # Break ties toward neurons the models were confident about.
            confidence = sum(float(p["scores"][n]) for p in probes) / len(probes)
            spread.append((len(distinct), confidence, n))
        spread.sort(reverse=True)
        return [n for _, _, n in spread[:args.n_neurons]]

    random.seed(args.seed)
    return sorted(random.sample(range(n_neurons), k=min(args.n_neurons, n_neurons)))


def plot_neuron(args, probes, neuron, out_path):
    rows, cols = len(probes), args.n_images
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.0 * rows), squeeze=False)

    for r, probe in enumerate(probes):
        for c in range(cols):
            ax = axes[r][c]
            ax.axis("off")
            if c < probe["top_ids"].shape[0]:
                image, _ = probe["data"][int(probe["top_ids"][c, neuron])]
                ax.imshow(image.resize([224, 224]))
        axes[r][0].set_title(
            "{}\n-> \"{}\"  ({:.3f})".format(probe["label"], probe["concepts"][neuron],
                                             float(probe["scores"][neuron])),
            loc="left", fontsize=11, pad=8)

    fig.suptitle("{} | {} | neuron {}".format(args.target_model, args.target_layer, neuron),
                 fontsize=13)
    # h_pad keeps each row's title clear of the images in the row above it; without
    # it the label sits on top of them, since the image axes have no visible frame.
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.5)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parser.parse_args()
    d_probes = args.d_probes.split(",")
    words = utils.load_concepts(args.concept_set)

    probes = [collect(args, d, words) for d in d_probes]

    sizes = {p["label"]: len(p["data"]) for p in probes}
    print("probing set sizes: {}".format(sizes))
    if max(sizes.values()) > 10 * min(sizes.values()):
        print("NOTE: these differ by more than 10x. CLIP-Dissect quality scales with the "
              "size and diversity of D_probe, so part of any difference you see below is "
              "probe-set size rather than probe-set content.")

    neurons = pick_neurons(args, probes)
    out_dir = os.path.join(args.out_dir, "{}_{}_{}".format(
        utils._clean_name(args.target_model), utils._clean_name(args.clip_model),
        args.target_layer))
    os.makedirs(out_dir, exist_ok=True)

    for neuron in neurons:
        path = os.path.join(out_dir, "neuron_{:04d}.png".format(neuron))
        plot_neuron(args, probes, neuron, path)
        summary = "  ".join('{}="{}"'.format(p["label"], p["concepts"][neuron])
                            for p in probes)
        print("neuron {:>5}  {}".format(neuron, summary))

    # A csv of every neuron's description under each probing set, so you can find
    # the interesting cases without paging through images.
    import pandas as pd
    n_neurons = min(p["n_neurons"] for p in probes)
    table = {"neuron": list(range(n_neurons))}
    for probe in probes:
        table["{}_concept".format(probe["label"])] = probe["concepts"][:n_neurons]
        table["{}_sim".format(probe["label"])] = probe["scores"][:n_neurons].numpy()
    csv_path = os.path.join(out_dir, "comparison.csv")
    pd.DataFrame(table).to_csv(csv_path, index=False)

    print("\nwrote {} figures + {} to {}".format(len(neurons), os.path.basename(csv_path), out_dir))


if __name__ == "__main__":
    main()
