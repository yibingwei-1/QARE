import logging
from dataclasses import dataclass
from transformers import ProcessorMixin
from src.arguments import DataArguments, ModelArguments
from src.model.processor import QWEN2_VL, process_vlm_inputs_fns

from src.utils import print_rank
from PIL import Image
import io

logger = logging.getLogger(__name__)


@dataclass
class MultimodalEvalDataCollator:
    processor: ProcessorMixin
    model_args: ModelArguments
    data_args: DataArguments
    encode_side: str

    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        texts, visual_inputs = [], []
        for example in batch:
            if example is None or not example:
                text, visual_input = '  ', None
            else:
                ex_text, ex_images = example[text_keyname], example[image_keyname]
                has_image = isinstance(ex_images, dict) or (isinstance(ex_images, list) and all(isinstance(item, dict) for item in ex_images))
                if has_image:
                    for text, raw_images in zip(ex_text, ex_images):
                        visual_input = []
                        assert 'resolutions' in raw_images, "Need len(raw_images['resolutions']) to determine the number of images"
                        num_images = len(raw_images['paths'])
                        for image_idx in range(num_images):
                            bytes_data = raw_images['bytes'][image_idx] if 'bytes' in raw_images else None
                            path = raw_images['paths'][image_idx] if 'paths' in raw_images else None
                            image_resolution = raw_images['resolutions'][image_idx] if 'resolutions' in raw_images else None
                            if bytes_data is None and path is None:
                                image = None
                            elif bytes_data is not None:
                                image = Image.open(io.BytesIO(bytes_data))
                            elif path is not None:
                                image = Image.open(path)
                            else:
                                image = None
                            if not self.data_args.resize_use_processor and image is not None and image_resolution:
                                image = image.resize(image_resolution)
                            visual_input.append(image)
                        texts.append(text)
                        visual_inputs.append(visual_input)
                else:
                    for text, visual_input in zip(ex_text, ex_images):
                        texts.append(text)
                        visual_inputs.append(visual_input)

        inputs = {'text': texts, 'images': visual_inputs}
        return inputs

    def __call__(self, examples):
        process_fn = process_vlm_inputs_fns[self.model_args.model_backbone]
        if self.encode_side == 'qry':
            assert type(examples[0]['query_text']) == list or type(examples[0]['query_image']) == list
            inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        else:
            assert type(examples[0]['cand_text']) == list or type(examples[0]['cand_image']) == list
            inputs = self._get_batch_inputs(examples, "cand_text", "cand_image")

        processed_inputs = process_fn(inputs, processor=self.processor, max_length=self.data_args.max_len)
        dataset_infos = [e["dataset_infos"] for e in examples]
        return processed_inputs, dataset_infos
