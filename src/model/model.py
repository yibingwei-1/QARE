from typing import Dict, Optional, Iterable
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from src.model.processor import QWEN2_VL, get_backbone_name

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info


class VLMModel(nn.Module):
    """
    Training-free VLM model for queryable attribute-specific embeddings (TF-QARE).

    Extracts attribute-disentangled representations by:
    1. Generating a short reply conditioned on an attribute-focused prompt
    2. Pooling hidden states from the penultimate decoder layer over reply tokens
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        use_reply_gen_pool: bool = True,
        reply_max_new_tokens: int = 64,
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config._attn_implementation = "flash_attention_2"
        if getattr(config, "vision_config", None) is not None:
            config.vision_config._attn_implementation = "flash_attention_2"

        model_backbone = get_backbone_name(hf_config=config)
        setattr(self, "model_backbone", model_backbone)

        self.encoder = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("cuda")

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.use_reply_gen_pool: bool = use_reply_gen_pool
        self.reply_max_new_tokens: int = reply_max_new_tokens

    def _get_decoder_layers(self):
        """Return the ModuleList of decoder blocks."""
        enc = getattr(self, "encoder", None)
        if enc is None:
            raise ValueError("`self.encoder` is not set.")

        try:
            layers = self.encoder.model.language_model.layers
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                return layers
        except AttributeError:
            pass

        for p in [
            "model.language_model.layers",
            "language_model.layers",
            "model.model.layers",
            "model.layers",
            "transformer.layers",
            "text_model.layers",
            "decoder.layers",
            "model.decoder.layers",
        ]:
            cur = self.encoder
            ok = True
            for name in p.split("."):
                if hasattr(cur, name):
                    cur = getattr(cur, name)
                else:
                    ok = False
                    break
            if ok and isinstance(cur, torch.nn.ModuleList) and len(cur) > 0:
                return cur

        raise ValueError("Cannot locate decoder `.layers` on the underlying language model.")

    def _resolve_layer_index(self, layer_index: int | None = None, layer_offset: int = -1) -> int:
        layers = self._get_decoder_layers()
        n = len(layers)
        if layer_index is not None:
            if not (0 <= layer_index < n):
                raise ValueError(f"layer_index={layer_index} out of range [0, {n-1}].")
            return layer_index
        idx = (n + layer_offset) if layer_offset < 0 else layer_offset
        if not (0 <= idx < n):
            raise ValueError(f"Resolved layer index {idx} out of range [0, {n-1}].")
        return idx

    def _register_hidden_state_hook(self, layer_index: int | None = None, layer_offset: int = -1):
        """Register a forward hook on a specific decoder layer to capture hidden states."""
        layers = self._get_decoder_layers()
        idx = self._resolve_layer_index(layer_index=layer_index, layer_offset=layer_offset)
        target = layers[idx]
        buffer: dict[str, torch.Tensor] = {}

        def _hook_fn(module, inputs, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            buffer["hidden"] = hs.detach()

        handle = target.register_forward_hook(_hook_fn)
        return handle, buffer, idx

    def _generate_reply_penultimate_pool_2stage(
        self,
        model_inputs: Dict[str, Tensor],
        max_new_tokens: Optional[int] = None,
        layer_offset: int = -2,
        exclude_eos: bool = True,
    ) -> Tensor:
        """
        Two-stage reply-token pooling from a specific decoder layer.

        Stage 1: Generate replies for the batch (no hidden states).
        Stage 2: Forward pass with a hook on the target layer; pool over reply tokens.

        Returns:
            (B, H) L2-normalized embeddings.
        """
        if max_new_tokens is None:
            max_new_tokens = int(getattr(self, "reply_max_new_tokens", 64))

        texts = model_inputs.get("texts", None)
        images = model_inputs.get("images", None)
        if texts is None or images is None:
            raise ValueError("Expected batched `texts` and `images` in model_inputs.")

        B = len(texts)
        if len(images) != B:
            raise ValueError("`texts` and `images` must have the same batch size.")

        batch_texts = []
        batch_image_inputs = []

        for i in range(B):
            prompt_text = texts[i].split("\n", 1)[1] if isinstance(texts[i], str) and ("\n" in texts[i]) else texts[i]

            imgs_i = images[i]
            if not isinstance(imgs_i, (list, tuple)):
                imgs_i = [imgs_i]

            content = [{"type": "image", "image": img} for img in imgs_i]
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]

            text_i = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            batch_texts.append(text_i)
            batch_image_inputs.append(image_inputs)

        inputs = self.processor(
            text=batch_texts,
            images=batch_image_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        src_lens = inputs["attention_mask"].sum(dim=1)

        # Stage 1: batched generation
        with torch.inference_mode():
            sequences = self.encoder.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=False,
            )

        # Stage 2: forward pass with hook
        tok = self.processor.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        eos_id = tok.eos_token_id

        replay = {
            "input_ids": sequences,
            "attention_mask": (sequences != pad_id).long(),
        }
        for k, v in inputs.items():
            if k not in ("input_ids", "attention_mask"):
                replay[k] = v

        handle, buffer, _ = self._register_hidden_state_hook(layer_index=None, layer_offset=layer_offset)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.autocast(device_type="cuda", dtype=torch.float16)
        )

        try:
            with torch.inference_mode(), amp_ctx:
                self.encoder(**replay, use_cache=False)
        finally:
            handle.remove()

        hidden = buffer.get("hidden", None)
        if hidden is None or not isinstance(hidden, torch.Tensor):
            raise RuntimeError("Hidden state hook did not capture any data.")

        B2, L, H = hidden.shape
        assert B2 == B, "Batch size mismatch."
        device = hidden.device

        pos = torch.arange(L, device=device).unsqueeze(0)
        src_lens = src_lens.to(device)
        seq_mask = (sequences != pad_id)
        reply_mask = (pos >= src_lens.unsqueeze(1)) & seq_mask
        if exclude_eos and eos_id is not None:
            reply_mask &= (sequences != eos_id)

        pooled_list = []
        for i in range(B):
            m = reply_mask[i]
            hi = hidden[i][m]
            if hi.numel() == 0:
                fb = max(int(src_lens[i].item()) - 1, 0)
                hi = hidden[i, fb:fb+1, :]
            hi_f = hi.float()
            hi_n = F.normalize(hi_f, p=2, dim=-1)
            pooled_i = hi_n.mean(dim=0, keepdim=True)
            pooled_list.append(pooled_i)

        pooled = torch.cat(pooled_list, dim=0)
        pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def encode_input(self, input):
        if getattr(self, "model_backbone", None) == QWEN2_VL:
            if getattr(self, "use_reply_gen_pool", False):
                texts = input.get("texts", None)
                images = input.get("images", None)
                if texts is not None and images is not None:
                    return self._generate_reply_penultimate_pool_2stage(
                        model_inputs=input,
                        max_new_tokens=64,
                        layer_offset=-2,
                        exclude_eos=True,
                    )

        raise NotImplementedError(f"Backbone {getattr(self, 'model_backbone', None)} not supported.")

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        qry_reps = self.encode_input(qry) if qry else None
        tgt_reps = self.encode_input(tgt) if tgt else None
        return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}
