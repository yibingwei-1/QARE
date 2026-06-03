from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "HuggingFace model name or path"})
    model_type: str = field(default=None, metadata={"help": "Model type override (usually auto-detected from config)"})
    model_backbone: str = field(default=None, metadata={"help": "Resolved backbone identifier"})
    checkpoint_path: str = field(default=None, metadata={"help": "Local model path"})
    pooling: str = field(default='last', metadata={"help": "Pooling method for encoder"})
    normalize: bool = field(default=False, metadata={"help": "Normalize representations"})
    temperature: float = field(default=0.02, metadata={"help": "Temperature for softmax"})
    num_crops: int = field(default=16, metadata={"help": "Number of crops for image encoder"})
    use_reply_gen_pool: bool = field(default=True, metadata={"help": "Use reply-generation pooling (TF-QARE)"})
    reply_max_new_tokens: int = field(default=64, metadata={"help": "Max new tokens for reply generation"})


@dataclass
class DataArguments:
    data_basedir: str = field(default=None, metadata={"help": "Base directory prepended to each dataset path if set"})
    eval_dataset_config: str = field(default=None, metadata={"help": "YAML file with evaluation dataset configuration"})
    encode_output_path: str = field(default=None, metadata={"help": "Output path for encoded embeddings"})
    max_len: int = field(default=None, metadata={"help": "Max input sequence length after tokenization"})
    image_resolution: str = field(default=None, metadata={"help": "Image resolution override"})
    resize_use_processor: bool = field(default=True, metadata={"help": "Resize via processor rather than custom code"})
    resize_min_pixels: int = field(default=28*28*4, metadata={"help": "Min pixels for image resize"})
    resize_max_pixels: int = field(default=28*28*1280, metadata={"help": "Max pixels for image resize"})
    image_decay_factor: float = field(default=None, metadata={"help": "Decay factor for temporal image resizing"})


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default=None, metadata={"help": "Output directory"})
    per_device_eval_batch_size: int = field(default=128, metadata={"help": "Batch size for evaluation"})
    dataloader_num_workers: int = field(default=4, metadata={"help": "Number of dataloader workers"})
    logging_steps: int = field(default=1, metadata={"help": "Logging steps"})
