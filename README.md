# TF-QARE: Training-Free Queryable Attribute-Specific Representation Extraction

**CVPR 2026 (Findings)**

> **Towards Text-Guided Attribute-Disentangled Multimodal Representation Learning**
>
> Yibing Wei, Sudeep Katakol, Manuel Brack, Jinhong Lin, Haoyue Bai, Yu-Teng Li, Richard Zhang, Eli Shechtman, Hareesh Ravi, Ajinkya Kale

TF-QARE extracts attribute-specific embeddings from frozen Vision-Language Models without any training. Given an image and a target attribute (object, style, or background), it produces an embedding that is sensitive to that attribute and invariant to others.

**Key idea:** Generate a short structured reply conditioned on an attribute-focused prompt, then pool hidden states from the penultimate decoder layer over the reply tokens.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # recommended for speed
```

## Quick Start

Extract an attribute-specific embedding from a single image:

```bash
python run_inference.py --image_path photo.jpg --attribute object
```

Compare two images on a specific attribute:

```bash
python run_inference.py --image_path img1.jpg --image_path_2 img2.jpg --attribute background
```

## Dataset

Download the QARE-Bench evaluation data from HuggingFace:

```bash
# TODO: Replace with actual HuggingFace dataset URL
# huggingface-cli download <HF_DATASET_URL> --local-dir ./data
```

Expected structure:
```
data/
├── AttriVerse/
│   ├── A/   (48 synthetic images: {object}_{background}_{style}.png)
│   └── B/   (18 synthetic images)
└── mission/
    ├── processed/
    │   ├── object_query_sets_v2.jsonl
    │   └── background_query_sets_v3.jsonl
    └── crops/
        ├── object_v2/   (SHA1-sharded JPGs)
        └── background/  (SHA1-sharded JPGs)
```

## Evaluation

Run QARE-Bench evaluation on the AttriVerse synthetic set:

```bash
python eval.py \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --eval_dataset_config configs/eval/attriverse.yaml \
    --encode_output_path outputs/attriverse \
    --per_device_eval_batch_size 1
```

Run on the Mission (real-world) dataset:

```bash
python eval.py \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --eval_dataset_config configs/eval/mission.yaml \
    --encode_output_path outputs/mission \
    --per_device_eval_batch_size 1
```

Multi-GPU evaluation:

```bash
torchrun --nproc_per_node=8 eval.py \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --eval_dataset_config configs/eval/attriverse.yaml \
    --encode_output_path outputs/attriverse \
    --per_device_eval_batch_size 1
```

## Results (QARE-Bench)

| Method | Synthetic mAP | Real mAP |
|--------|:---:|:---:|
| CLIP | 4.5 | 27.7 |
| SigLIP | 4.4 | 28.8 |
| DINOv2 | 4.2 | 27.7 |
| VLM2Vec-V1 (Qwen2-VL-7B) | 16.7 | 36.2 |
| VLM2Vec-V2 (Qwen2-VL-2B) | 15.4 | 45.5 |
| **TF-QARE (Qwen2-VL-7B)** | **78.4** | **64.3** |

## Citation

```bibtex
@inproceedings{wei2026towards,
    title={Towards Text-Guided Attribute-Disentangled Multimodal Representation Learning},
    author={Wei, Yibing and Katakol, Sudeep and Brack, Manuel and Lin, Jinhong and Bai, Haoyue and Li, Yu-Teng and Zhang, Richard and Shechtman, Eli and Ravi, Hareesh and Kale, Ajinkya},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    year={2026}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
