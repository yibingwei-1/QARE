import os
import json

from datasets import Dataset
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING
from src.model.processor import process_input_text


ATTRIBUTES = ['object', 'background']

QUERY_TEXT_DICT = {
    "object": "<|image_1|>\nDescribe ONLY the main object in the image using a two-part structured format.\nFORMAT MUST MATCH EXACTLY:\nMain object is [main summary], [detailed description].\nRules:\n- [main summary]: 1-3 words describing the object's category, color, and general appearance.\n- [detailed description]: 15-30 words expanding on shape, material, surface, parts, or posture.\n- Focus on ONE main foreground object only; ignore background, scene, or style.\n- Write up to TWO sentences; no lists, no line breaks, no quotes.\n",
    "background": "<|image_1|>\nDescribe ONLY the background environment in the image using a two-part structured format.\nFORMAT MUST MATCH EXACTLY:\nBackground is [main summary], [detailed description].\nRules:\n- [main summary]: 1-3 words describing the setting type and general atmosphere.\n- [detailed description]: 15-30 words expanding on spatial layout, lighting conditions, colors, and environmental details.\n- Focus ONLY on the background environment; ignore the main subject/object and artistic style.\n- Write up to TWO sentences; no lists, no line breaks, no quotes.\n",
}
TARGET_TEXT_DICT = QUERY_TEXT_DICT


def _resolve_object_rel_path(base_name):
    if not base_name:
        return ""
    prefix = base_name[:2] if len(base_name) >= 2 else "00"
    return os.path.join("crops", "object_v2", prefix, base_name)


def _resolve_background_rel_path(base_name):
    if not base_name:
        return ""
    prefix = base_name[:2] if len(base_name) >= 2 else "00"
    return os.path.join("crops", "background", prefix, base_name)


def _load_object_split(jsonl_path_obj):
    samples = []
    if not os.path.exists(jsonl_path_obj):
        return Dataset.from_list(samples)

    with open(jsonl_path_obj, "r") as f:
        for line in f:
            row = json.loads(line.strip())
            query = row.get("query", {}) or {}
            query_base = query.get("crop_filename", "")
            if not query_base:
                continue

            pos_map = row.get("positives", {}) or {}
            positives = list(pos_map.values())
            negatives = list(row.get("negatives", []) or [])

            if len(positives) == 0 and len(negatives) == 0:
                continue

            positives.sort()
            negatives.sort()

            qry_img_rel = _resolve_object_rel_path(query_base)
            pos_rel_paths = [_resolve_object_rel_path(p) for p in positives]
            neg_rel_paths = [_resolve_object_rel_path(n) for n in negatives]

            num_images = len(pos_rel_paths) + len(neg_rel_paths)
            other_images = pos_rel_paths + neg_rel_paths
            other_images = other_images * len(ATTRIBUTES)

            for attr in ATTRIBUTES:
                sample = {
                    'qry_inst': QUERY_TEXT_DICT[attr],
                    'qry_text': '',
                    'qry_img_path': qry_img_rel,
                    'tgt_inst': '',
                    'tgt_text': [TARGET_TEXT_DICT['object']] * num_images + [TARGET_TEXT_DICT['background']] * num_images,
                    'tgt_img_path': other_images,
                    'gt_img_path': pos_rel_paths,
                    'query_attr': attr,
                    'tgt_attr': ['object'] * num_images + ['background'] * num_images,
                }
                samples.append(sample)

    return Dataset.from_list(samples)


def _load_background_split(jsonl_path_bg):
    samples = []
    if not os.path.exists(jsonl_path_bg):
        return Dataset.from_list(samples)

    with open(jsonl_path_bg, "r") as f:
        for line in f:
            row = json.loads(line.strip())
            query = row.get("query", {}) or {}
            query_base = query.get("image_base", "")
            if not query_base:
                continue

            positives = list(row.get("positives", []) or [])
            negatives = list(row.get("negatives", []) or [])

            if len(positives) == 0 and len(negatives) == 0:
                continue

            positives.sort()
            negatives.sort()

            qry_img_rel = _resolve_background_rel_path(query_base)
            pos_rel_paths = [_resolve_background_rel_path(p) for p in positives]
            neg_rel_paths = [_resolve_background_rel_path(n) for n in negatives]

            num_images = len(pos_rel_paths) + len(neg_rel_paths)
            other_images = pos_rel_paths + neg_rel_paths
            other_images = other_images * len(ATTRIBUTES)

            for attr in ATTRIBUTES:
                sample = {
                    'qry_inst': QUERY_TEXT_DICT[attr],
                    'qry_text': '',
                    'qry_img_path': qry_img_rel,
                    'tgt_inst': '',
                    'tgt_text': [TARGET_TEXT_DICT['object']] * num_images + [TARGET_TEXT_DICT['background']] * num_images,
                    'tgt_img_path': other_images,
                    'gt_img_path': pos_rel_paths,
                    'query_attr': attr,
                    'tgt_attr': ['object'] * num_images + ['background'] * num_images,
                }
                samples.append(sample)

    return Dataset.from_list(samples)


def _load_mission_from_jsonl(obj_jsonl_path, bg_jsonl_path):
    object_ds = _load_object_split(obj_jsonl_path)
    background_ds = _load_background_split(bg_jsonl_path)

    rows = []
    if object_ds is not None and len(object_ds) > 0:
        rows.extend(list(object_ds))
    if background_ds is not None and len(background_ds) > 0:
        rows.extend(list(background_ds))

    dataset = Dataset.from_list(rows) if len(rows) > 0 else Dataset.from_list([])
    return dataset


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs.get('image_root', './data/mission/')

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for qry_inst, qry_text, qry_img_rel, tgt_inst, tgt_attrs, tgt_captions, tgt_img_paths, gt_img_paths, query_attr in (
            zip(batch_dict['qry_inst'], batch_dict['qry_text'], batch_dict['qry_img_path'],
            batch_dict['tgt_inst'], batch_dict['tgt_attr'], batch_dict['tgt_text'], batch_dict['tgt_img_path'], batch_dict['gt_img_path'], batch_dict['query_attr'])):
        qry_inst = "\n" + qry_inst.replace("<|image_1|>", "").strip()
        qry_text = process_input_text(qry_inst, model_backbone, text=qry_text, add_image_token=True)
        qry_text = qry_text.replace(" \n", "\n") + "\n"
        query_texts.append([qry_text])

        qry_img_path = os.path.join(image_root, qry_img_rel)
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


DATASET_PARSER_NAME = "mission"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_mission_dataset(model_args, data_args, *args, **kwargs):
    image_root = kwargs.get('image_root', './data/mission/')
    obj_jsonl_path = kwargs.get('obj_jsonl_path', "object_query_sets_v2.jsonl")
    bg_jsonl_path = kwargs.get('bg_jsonl_path', "background_query_sets_v3.jsonl")
    obj_jsonl_path = os.path.join(image_root, "processed", obj_jsonl_path)
    bg_jsonl_path = os.path.join(image_root, "processed", bg_jsonl_path)

    dataset = _load_mission_from_jsonl(obj_jsonl_path, bg_jsonl_path)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['image_root'] = image_root

    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    return dataset, None
