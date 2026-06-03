import os
import sys

from datasets import Dataset
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING
from src.model.processor import process_input_text


# Define the three attributes and their corresponding templates
ATTRIBUTES = ['object', 'background','style']

# QUERY_TEXT_DICT = {
#     "object": "<|image_1|>\nFind an image that contains the same main object as this one, while ignoring its style and background.\n",
#     "background": "<|image_1|>\nFind an image with a similar background to this one, while ignoring the main subject and style.\n",
#     "style": "<|image_1|>\nFind an image with a similar visual style to this one, while ignoring the depicted content.\n",
# }

# TARGET_TEXT_DICT = {
#     "object": '<|image_1|>\nRepresent the main subject of the given image, ignoring the style and background in it.\n',
#     "background": '<|image_1|>\nRepresent the background of the given image, ignoring the main subject and style in it.\n',
#     "style": '<|image_1|>\nRepresent the style of the given image, ignoring the main subject and content in it.\n',
# }

# QUERY_TEXT_DICT = {
#     "object": "<|image_1|>\nIdentify only the main object in the image. Output just a short noun phrase or single word (e.g., 'car', 'wooden chair', 'red apple'). Do NOT describe any actions, background, or scene context. Output format: 'The main object is [object]'.\n",
    
#     "background": "<|image_1|>\nIdentify only the background environment of the image. Output just a short noun phrase or single word (e.g., 'forest', 'city street', 'beach'). Do NOT mention the main subject or visual style. Output format: 'The background is [background]'.\n",
    
#     "style": "<|image_1|>\nIdentify only the visual style of the image. Output just a short list of adjectives or short noun phrases (e.g., 'soft lighting', 'vibrant colors', 'minimalist design'). Focus on tone, color, texture, lighting, composition, and mood — not subjects or content. Do NOT mention any objects, people, animals, locations, or actions. Output format: 'The visual style is [style]'.\n",
# }

QUERY_TEXT_DICT = {
    "object": "<|image_1|>\nDescribe ONLY the main object in the image using a two-part structured format.\nFORMAT MUST MATCH EXACTLY:\nMain object is [main summary], [detailed description].\nRules:\n- [main summary]: 1-3 words describing the object's category, color, and general appearance.\n- [detailed description]: 15-30 words expanding on shape, material, surface, parts, or posture.\n- Focus on ONE main foreground object only; ignore background, scene, or style.\n- Write up to TWO sentences; no lists, no line breaks, no quotes.\n",

    "background": "<|image_1|>\nDescribe ONLY the background environment in the image using a two-part structured format.\nFORMAT MUST MATCH EXACTLY:\nBackground is [main summary], [detailed description].\nRules:\n- [main summary]: 1-3 words describing the setting type and general atmosphere.\n- [detailed description]: 15-30 words expanding on spatial layout, lighting conditions, colors, and environmental details.\n- Focus ONLY on the background environment; ignore the main subject/object and artistic style.\n- Write up to TWO sentences; no lists, no line breaks, no quotes.\n",

    "style": "<|image_1|>\nDescribe ONLY the visual style of the image using a two-part structured format.\nFORMAT MUST MATCH EXACTLY:\nVisual style is [main summary], [detailed description].\nRules:\n- [main summary]: 1-3 words describing the overall artistic medium or rendering technique.\n- [detailed description]: 15-30 words expanding on color palette, lighting treatment, texture quality, contrast, and composition mood.\n- Focus ONLY on visual style and rendering; ignore objects, people, animals, locations, or actions.\n- Write up to TWO sentences; no lists, no line breaks, no quotes.\n",
}



# QUERY_TEXT_DICT = {
#     "object": "<|image_1|>\nIdentify the main object in the image. Ignore background and context. Output key points describing the object briefly. Format: '- [key point]'.",

#     "background": "<|image_1|>\nIdentify the background environment of the image. Ignore the main subject. Output key points describing the setting. Format: '- [key point]'.",

#     "style": "<|image_1|>\nDescribe the image’s visual style in key points focusing on tone, color, lighting, and mood. Do not mention objects or actions. Begin with: '- [key point]'."
# }



TARGET_TEXT_DICT = QUERY_TEXT_DICT

def find_gt_img_path(query_attr, tgt_attr_idx, other_images):
    gt_img_path = []
    for image_file in other_images:
        filename_parts = image_file.replace('.png', '').split('_')
        if query_attr == filename_parts[tgt_attr_idx]:
            gt_img_path.append(image_file)
    return gt_img_path

def _load_attriverse_viz_dataset_from_dir(dataset_root):
    # Get all PNG files and sort them for deterministic behavior
    image_files = [f for f in os.listdir(dataset_root) if f.endswith('.png')]
    image_files.sort()
    
    # Create samples
    samples = []
    
    for image_file in image_files:
        # Parse filename: <object>_<background>_<style>.png
        filename_parts = image_file.replace('.png', '').split('_')
        if len(filename_parts) != 3:
            continue  # Skip files that don't follow the expected format
        
        # object_name, background_name, style_name = filename_parts
        
        # Get all other images in the split
        other_images = image_files * 3
        num_images = len(image_files)
        
        # Create one sample for each attribute
        for attr_idx, attr in enumerate(ATTRIBUTES):
            # Construct relative paths for all other images
            gt_img_paths = find_gt_img_path(filename_parts[attr_idx], attr_idx, image_files)
            
            sample = {
                'qry_inst': QUERY_TEXT_DICT[attr],
                'qry_text': '',  # Always empty string
                'qry_img_path': image_file,
                'tgt_inst': '',
                'tgt_text': [TARGET_TEXT_DICT['object']] * num_images + [TARGET_TEXT_DICT['background']] * num_images + [TARGET_TEXT_DICT['style']] * num_images,  # One empty string per candidate
                'tgt_img_path': other_images,
                'gt_img_path': gt_img_paths,
                'query_attr': attr,
                'tgt_attr': ['object'] * num_images + ['background'] * num_images + ['style'] * num_images,
            }
            
            samples.append(sample)
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(samples)
    return dataset


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs['image_root']

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for qry_inst, qry_text, qry_img_path, tgt_inst, tgt_attrs, tgt_captions, tgt_img_paths, gt_img_paths, query_attr in (
            zip(batch_dict['qry_inst'], batch_dict['qry_text'], batch_dict['qry_img_path'], 
            batch_dict['tgt_inst'], batch_dict['tgt_attr'], batch_dict['tgt_text'], batch_dict['tgt_img_path'], batch_dict['gt_img_path'], batch_dict['query_attr'])):
        qry_inst = "\n" + qry_inst.replace("<|image_1|>", "").strip()
        qry_text = process_input_text(qry_inst, model_backbone, text=qry_text, add_image_token=True)
        # to stay consistent with v1 eval
        qry_text = qry_text.replace(" \n", "\n") + "\n"
        query_texts.append([qry_text])

        qry_img_path = os.path.join(image_root, qry_img_path)
        query_images.append([{"bytes": [None], "paths": [qry_img_path],
                            "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}])
        
        tgt_inst_captions = []
        for tgt_cap in tgt_captions:
            tgt_cap = tgt_cap.replace("<|image_1|>", "")
            tgt_inst_caption = process_input_text(tgt_cap, model_backbone, text='', add_image_token=True)
            tgt_inst_caption = tgt_inst_caption.replace(" \n", "\n")
            tgt_inst_captions.append(tgt_inst_caption)
        cand_texts.append(tgt_inst_captions)

        cand_img_paths = [os.path.join(image_root, tgt_img_path) for tgt_img_path in tgt_img_paths]
        img_list = [{"bytes": [None], "paths": [cand_img_path],
                     "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]} for cand_img_path in cand_img_paths]
        cand_images.append(img_list)

        cand_names = [path+':'+a.strip('"') for path, a in zip(tgt_img_paths, tgt_attrs)]

        dataset_infos.append({
            "cand_names": cand_names,
            "cand_attr": tgt_attrs,
            "label_name": [path+':'+query_attr.strip('"') for path in gt_img_paths],
            "query_attr": query_attr,
            "qry_img_path": qry_img_path,
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "attriverse_viz"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_attriverse_viz_dataset(model_args, data_args, *args, **kwargs):
    dataset = _load_attriverse_viz_dataset_from_dir(kwargs['image_root'])

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    return dataset, None
