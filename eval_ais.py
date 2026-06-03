"""
Compute Average Intra-image Similarity (AIS) for QARE-Bench.

AIS measures how much an embedding model disentangles attributes. For each image,
we extract embeddings conditioned on each attribute (object, background, style),
then compute the mean pairwise cosine similarity between them.

Lower AIS = better disentanglement (different attributes produce different embeddings).

Usage:
    # Single GPU
    python eval_ais.py \
        --model_name Qwen/Qwen2-VL-7B-Instruct \
        --image_root ./data/AttriVerse/A/ \
        --output_path outputs/ais_results.json

    # Multi-GPU
    torchrun --nproc_per_node=8 eval_ais.py \
        --model_name Qwen/Qwen2-VL-7B-Instruct \
        --image_root ./data/AttriVerse/A/ \
        --output_path outputs/ais_results.json
"""
import argparse
import datetime
import json
import os
import sys
import time
import random

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model.model import VLMModel


ATTRIBUTE_PROMPTS = {
    "object": "Return OBJ=<ONE_WORD_NOUN>.\nRules: one lemma in ALLCAPS (e.g., CAR, APPLE). No background, no actions, no style. No quotes.",
    "background": "Return BKG#[scene_label].\nRules: use lowercase_snake_case Places/scene term (e.g., city_street, sandy_beach, forest_trail). No objects, no style, no actions. No quotes.",
    "style": "Return STY::{adj1;adj2;adj3}.\nRules: three lowercase adjectives about tone/color/texture/lighting/mood only (e.g., soft;high_contrast;filmic). No objects, no places, no actions. No quotes.",
}

ATTRIBUTES = list(ATTRIBUTE_PROMPTS.keys())


def compute_ais(embeddings_per_image: dict) -> dict:
    """
    Compute AIS given a dict mapping image_name -> {attr: embedding_vector}.

    Returns per-image AIS values and the overall mean.
    """
    ais_values = []

    for img_name, attr_embeds in embeddings_per_image.items():
        vecs = []
        for attr in ATTRIBUTES:
            if attr in attr_embeds:
                v = attr_embeds[attr].astype(np.float64)
                v = v / (np.linalg.norm(v) + 1e-12)
                vecs.append(v)

        n = len(vecs)
        if n < 2:
            continue

        sim_sum = 0.0
        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim_sum += float(np.dot(vecs[i], vecs[j]))
                pair_count += 1

        ais_values.append(sim_sum / pair_count)

    return {
        "AIS_mean": float(np.mean(ais_values)) if ais_values else 0.0,
        "AIS_std": float(np.std(ais_values)) if ais_values else 0.0,
        "num_images": len(ais_values),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute AIS (Average Intra-image Similarity)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--image_root", type=str, required=True, help="Directory with images (e.g., ./data/AttriVerse/A/)")
    parser.add_argument("--output_path", type=str, default="outputs/ais_results.json")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    # DDP setup
    if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if local_rank == 0:
        print(f"Loading model: {args.model_name}")

    # DDP-safe model loading
    if local_rank == 0:
        model = VLMModel(model_name=args.model_name, use_reply_gen_pool=True, reply_max_new_tokens=args.max_new_tokens)
    if dist.is_initialized():
        dist.barrier()
    if local_rank != 0:
        model = VLMModel(model_name=args.model_name, use_reply_gen_pool=True, reply_max_new_tokens=args.max_new_tokens)
        time.sleep(random.randint(2 * local_rank, 3 * local_rank))
    model.eval()

    # Gather all images
    image_files = sorted([f for f in os.listdir(args.image_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if local_rank == 0:
        print(f"Found {len(image_files)} images in {args.image_root}")

    # Build work items: (image_file, attribute)
    work_items = [(img, attr) for img in image_files for attr in ATTRIBUTES]

    # Split work across ranks
    if dist.is_initialized():
        items_per_rank = len(work_items) // world_size
        remainder = len(work_items) % world_size
        start = local_rank * items_per_rank + min(local_rank, remainder)
        end = start + items_per_rank + (1 if local_rank < remainder else 0)
        local_items = work_items[start:end]
    else:
        local_items = work_items

    # Encode
    local_results = []
    for img_file, attr in tqdm(local_items, desc=f"AIS encoding (rank {local_rank})", disable=local_rank > 0):
        img_path = os.path.join(args.image_root, img_file)
        img = Image.open(img_path).convert("RGB")
        prompt = ATTRIBUTE_PROMPTS[attr]

        inputs = {"texts": [prompt], "images": [[img]]}
        with torch.no_grad():
            emb = model.encode_input(inputs).cpu().numpy()[0]

        local_results.append((img_file, attr, emb))

    # Gather results
    if dist.is_initialized():
        all_results_gathered = [None for _ in range(world_size)]
        dist.all_gather_object(all_results_gathered, local_results)
        all_results = [item for rank_results in all_results_gathered for item in rank_results]
    else:
        all_results = local_results

    # Compute AIS (rank 0 only)
    if local_rank == 0:
        embeddings_per_image = {}
        for img_file, attr, emb in all_results:
            if img_file not in embeddings_per_image:
                embeddings_per_image[img_file] = {}
            embeddings_per_image[img_file][attr] = emb

        results = compute_ais(embeddings_per_image)
        print(f"\n{'='*50}")
        print(f"AIS Results ({len(image_files)} images, {len(ATTRIBUTES)} attributes)")
        print(f"{'='*50}")
        print(f"AIS (mean): {results['AIS_mean']:.4f}")
        print(f"AIS (std):  {results['AIS_std']:.4f}")
        print(f"{'='*50}")

        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
