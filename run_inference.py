"""
Standalone TF-QARE inference: extract attribute-specific embeddings from a single image.

Usage:
    python run_inference.py --image_path photo.jpg --attribute object
    python run_inference.py --image_path img1.jpg --image_path_2 img2.jpg --attribute background
"""
import argparse
import torch
import numpy as np
from src.model.model import VLMModel


ATTRIBUTE_PROMPTS = {
    "object": "Describe ONLY the main object in the image using a two-part structured format.\nFORMAT MUST MATCH EXACTLY:\nMain object is [main summary], [detailed description].\nRules:\n- [main summary]: 1-3 words describing the object's category, color, and general appearance.\n- [detailed description]: 15-30 words expanding on shape, material, surface, parts, or posture.\n- Focus on ONE main foreground object only; ignore background, scene, or style.\n- Write up to TWO sentences; no lists, no line breaks, no quotes.",
    "background": "Describe ONLY the background environment in the image using a two-part structured format.\nFORMAT MUST MATCH EXACTLY:\nBackground is [main summary], [detailed description].\nRules:\n- [main summary]: 1-3 words describing the setting type and general atmosphere.\n- [detailed description]: 15-30 words expanding on spatial layout, lighting conditions, colors, and environmental details.\n- Focus ONLY on the background environment; ignore the main subject/object and artistic style.\n- Write up to TWO sentences; no lists, no line breaks, no quotes.",
    "style": "Describe ONLY the visual style of the image using a two-part structured format.\nFORMAT MUST MATCH EXACTLY:\nVisual style is [main summary], [detailed description].\nRules:\n- [main summary]: 1-3 words describing the overall artistic medium or rendering technique.\n- [detailed description]: 15-30 words expanding on color palette, lighting treatment, texture quality, contrast, and composition mood.\n- Focus ONLY on visual style and rendering; ignore objects, people, animals, locations, or actions.\n- Write up to TWO sentences; no lists, no line breaks, no quotes.",
}


def main():
    parser = argparse.ArgumentParser(description="TF-QARE: Extract attribute-specific embeddings")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the query image")
    parser.add_argument("--image_path_2", type=str, default=None, help="Optional second image for similarity comparison")
    parser.add_argument("--attribute", type=str, required=True, choices=["object", "style", "background"])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    model = VLMModel(
        model_name=args.model_name,
        use_reply_gen_pool=True,
        reply_max_new_tokens=args.max_new_tokens,
    )
    model.eval()

    from PIL import Image
    prompt = ATTRIBUTE_PROMPTS[args.attribute]

    img1 = Image.open(args.image_path).convert("RGB")
    input_1 = {"texts": [prompt], "images": [[img1]]}

    with torch.no_grad():
        emb1 = model.encode_input(input_1).cpu().numpy()

    print(f"\nAttribute: {args.attribute}")
    print(f"Image: {args.image_path}")
    print(f"Embedding shape: {emb1.shape}")
    print(f"First 5 dims: {emb1[0, :5]}")

    if args.image_path_2:
        img2 = Image.open(args.image_path_2).convert("RGB")
        input_2 = {"texts": [prompt], "images": [[img2]]}
        with torch.no_grad():
            emb2 = model.encode_input(input_2).cpu().numpy()

        similarity = float(np.dot(emb1[0], emb2[0]))
        print(f"\nImage 2: {args.image_path_2}")
        print(f"Cosine similarity ({args.attribute}): {similarity:.4f}")


if __name__ == "__main__":
    main()
