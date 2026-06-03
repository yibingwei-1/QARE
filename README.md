# TF-QARE: Training-Free Queryable Attribute-Specific Representation Extraction

**CVPR 2026 (Findings)**

> **Towards Text-Guided Attribute-Disentangled Multimodal Representation Learning**
>
> Yibing Wei, Sudeep Katakol, Manuel Brack, Jinhong Lin, Haoyue Bai, Yu-Teng Li, Richard Zhang, Eli Shechtman, Hareesh Ravi, Ajinkya Kale

[📄 Paper](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wei_Towards_Text-Guided_Attribute-Disentangled_Multimodal_Representation_Learning_CVPRF_2026_paper.pdf) | [🤗 Dataset](https://huggingface.co/datasets/sudeepk/QARE-Bench)

TF-QARE extracts attribute-specific embeddings from frozen Vision-Language Models without any training. Given an image and a target attribute (object, style, or background), it produces an embedding that is sensitive to that attribute and invariant to others.

**Key idea:** Given an image and an attribute-focused prompt, TF-QARE generates an attribute-specific reply and pools the penultimate-layer hidden states over the reply tokens.

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

Download the QARE-Bench evaluation data from [HuggingFace](https://huggingface.co/datasets/sudeepk/QARE-Bench):

```bash
huggingface-cli download sudeepk/QARE-Bench --repo-type dataset --local-dir ./data
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

TF-QARE reports two complementary metrics:

* **mAP** measures attribute-conditioned retrieval: whether images are retrieved according to the queried attribute.
* **AIS** measures intra-image disentanglement: whether object / style / background embeddings from the same image are separated in representation space.

### Prompt configurations

The retrieval and AIS evaluations use different prompt configurations because they measure different properties:

| Metric | Prompt configuration                  | Max new tokens | Purpose                          |
| ------ | ------------------------------------- | :------------: | -------------------------------- |
| mAP    | Structured two-part attribute prompts |       64       | Attribute-conditioned retrieval  |
| AIS    | Compact keyword-style prompts         |       16       | Intra-image attribute separation |

The structured prompts are designed to produce descriptive attribute-specific replies for retrieval. The compact AIS prompts are designed to reduce shared response structure across attributes when measuring intra-image embedding similarity.

Run QARE-Bench retrieval evaluation on the released synthetic set:

```bash
python eval.py \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --eval_dataset_config configs/eval/attriverse.yaml \
    --encode_output_path outputs/attriverse \
    --per_device_eval_batch_size 1
```

Run AIS evaluation:

```bash
python eval_ais.py \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --eval_dataset_config configs/eval/attriverse.yaml \
    --encode_output_path outputs/attriverse_ais \
    --max_new_tokens 16
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

| Method                    | Synthetic mAP ↑ | Synthetic AIS ↓ | Real mAP ↑ | Real AIS ↓ |
| ------------------------- | :-------------: | :-------------: | :--------: | :--------: |
| CLIP                      |       4.5       |       1.00      |    27.7    |    1.00    |
| SigLIP                    |       4.4       |       1.00      |    28.8    |    1.00    |
| DINOv2                    |       4.2       |       1.00      |    27.7    |    1.00    |
| VLM2Vec-V1 (Qwen2-VL-7B)  |       16.7      |       0.97      |    36.2    |    0.96    |
| VLM2Vec-V2 (Qwen2-VL-2B)  |       15.4      |       0.82      |    45.5    |    0.81    |
| **TF-QARE (Qwen2-VL-7B)** |     **78.4**    |     **0.68**    |  **64.3**  |  **0.59**  |

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
