import argparse
import torch

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import json
from tqdm import tqdm
from collections import defaultdict

import os
os.makedirs("results/mix-mplug-owl-2-boost/", exist_ok=True)



def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    
    
    import json

    
    image_paths = [
        "../datasets/LIVEC/Images/",
        "../datasets/AGIQA-3K/database/",
        "../datasets/1024x768/",
        "../datasets/SPAQ/",
        "../datasets/FLIVE_Database/database/",
        "../datasets/CGIQA-6K/database/",
        "../datasets/kadid10k/images/",
    ]

    json_prefix = "../datasets/json/"
    jsons = [
        json_prefix + "livec.json",
        json_prefix + "agi.json",
        json_prefix + "koniq.json",
        json_prefix + "spaq.json",
        json_prefix + "flive.json",
        json_prefix + "cgi.json",
        json_prefix + "kadid.json",
    ]

    conv_mode = "mplug_owl2"
    
    inp = "Rate the quality of the image."
        
    conv = conv_templates[conv_mode].copy()
    inp = DEFAULT_IMAGE_TOKEN + inp
    conv.append_message(conv.roles[0], inp)
    image = None
        
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    toks = ["good", "poor", "high", "low", "excellent", "bad", "fine", "moderate",  "decent", "average", "medium", "acceptable"]
    print(toks)
    ids_ = [id_[1] for id_ in tokenizer(toks)["input_ids"]]
    print(ids_)
    
    for image_path, json_ in zip(image_paths, jsons):
        with open(json_) as f:
            iqadata = json.load(f)  

        
        
        for i, llddata in enumerate(tqdm(iqadata, desc="Evaluating [{}]".format(json_.split("/")[-1]))):
            #print(f"Evaluating image {i}")
            #print(prompt)
            filename = llddata["img_path"]
            llddata["logits"] = defaultdict(float)

            image = load_image(image_path + filename)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_logits = model(input_ids,
                    images=image_tensor)["logits"][:,-1]

            for tok, id_ in zip(toks, ids_):
                llddata["logits"][tok] += output_logits[0,id_].item()
                    
            with open(f"results/mix-mplug-owl-2-boost/{json_.split('/')[-1]}", "a") as wf:
                json.dump(llddata, wf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="teowu/mplug_owl2_7b_448_qinstruct_preview_v0.1")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)