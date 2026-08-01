import os
import argparse
import datetime
import json
import pandas as pd
import torch

import utils
import similarity


parser = argparse.ArgumentParser(description='CLIP-Dissect')

parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                    help="""Which model scores the concepts. OpenAI CLIP: RN50, RN101, RN50x4,
                         RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14. open_clip: bioclip,
                         bioclip2, hf-hub:org/repo, or open_clip:ARCH:PRETRAINED. Use bioclip or
                         bioclip2 when the concept set is biological -- OpenAI CLIP's text encoder
                         has seen very few taxonomic names.""")
parser.add_argument("--target_model", type=str, default="resnet50",
                   help=""""Which model to dissect. Pretrained imagenet models from torchvision,
                        resnet18_places, the timm baselines vit_in21k/dino_vitb16, or a CLIP-style
                        image tower: bioclip, bioclip2, clip_ViT-B/16, hf-hub:org/repo,
                        open_clip:ARCH:PRETRAINED""")
parser.add_argument("--target_layers", type=str, default="conv1,layer1,layer2,layer3,layer4",
                    help="""Which layer neurons to describe. String list of layer names to describe, separated by comma(no spaces).
                          Follows the naming scheme of the Pytorch module used. For CLIP-style image
                          towers the names are block_0..block_{L-1}, out and proj -- run
                          `python list_layers.py --target_model <name>` to see them.""")
parser.add_argument("--d_probe", type=str, default="broden",
                    help="""Probing dataset. A name registered in data_utils.DATASET_ROOTS
                         (imagenet_val, broden, imagenet_broden, cifar10/100_train/val,
                         places365_train/val, treeoflife_*, birds525_*), or
                         imagefolder:/path/to/dir for a directory of class subfolders.""")
parser.add_argument("--concept_set", type=str, default="data/20k.txt", help="Path to txt file containing concept set")
parser.add_argument("--prompt_template", type=str, default="{}",
                    help="""Template applied to each concept before encoding, must contain '{}'.
                         Defaults to the bare concept. Try 'a photo of {}.' with BioCLIP.""")
parser.add_argument("--batch_size", type=int, default=200, help="Batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--activation_dir", type=str, default="saved_activations", help="where to save activations")
parser.add_argument("--result_dir", type=str, default="results", help="where to save results")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")
parser.add_argument("--similarity_fn", type=str, default="soft_wpmi", choices=["soft_wpmi", "wpmi", "rank_reorder", 
                                                                               "cos_similarity", "cos_similarity_cubed"])

parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()
    args.target_layers = args.target_layers.split(",")
    
    similarity_fn = eval("similarity.{}".format(args.similarity_fn))
    
    utils.save_activations(clip_name = args.clip_model, target_name = args.target_model,
                           target_layers = args.target_layers, d_probe = args.d_probe,
                           concept_set = args.concept_set, batch_size = args.batch_size,
                           device = args.device, pool_mode=args.pool_mode,
                           save_dir = args.activation_dir,
                           prompt_template = args.prompt_template)

    outputs = {"layer":[], "unit":[], "description":[], "similarity":[]}
    # Same loader save_activations used, so index i here is row i of the text features.
    words = utils.load_concepts(args.concept_set)

    for target_layer in args.target_layers:
        save_names = utils.get_save_names(clip_name = args.clip_model, target_name = args.target_model,
                                  target_layer = target_layer, d_probe = args.d_probe,
                                  concept_set = args.concept_set, pool_mode = args.pool_mode,
                                  save_dir = args.activation_dir)
        target_save_name, clip_save_name, text_save_name = save_names

        similarities = utils.get_similarity_from_activations(
            target_save_name, clip_save_name, text_save_name, similarity_fn, return_target_feats=False, device=args.device
        )
        vals, ids = torch.max(similarities, dim=1)
        
        del similarities
        torch.cuda.empty_cache()
        
        descriptions = [words[int(idx)] for idx in ids]
        
        outputs["unit"].extend([i for i in range(len(vals))])
        outputs["layer"].extend([target_layer]*len(vals))
        outputs["description"].extend(descriptions)
        outputs["similarity"].extend(vals.cpu().numpy())
        
    df = pd.DataFrame(outputs)
    # The probing model is part of the identity of a run now that it is not always
    # OpenAI CLIP, so it goes in the directory name alongside the target.
    save_path = "{}/{}_{}_{}".format(args.result_dir,
                                     utils._clean_name(args.target_model),
                                     utils._clean_name(args.clip_model),
                                     datetime.datetime.now().strftime("%y_%m_%d_%H_%M"))
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path,"descriptions.csv"), index=False)
    with open(os.path.join(save_path, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)