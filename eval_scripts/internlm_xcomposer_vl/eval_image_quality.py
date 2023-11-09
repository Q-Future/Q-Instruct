from transformers import AutoModel, AutoTokenizer
import torch

import random
from copy import deepcopy
from PIL import Image
import json
from tqdm import tqdm

torch.set_grad_enabled(False)


import os
os.makedirs("results/mix-internlm_xcomposer_vl/",exist_ok=True)
torch.manual_seed(1234)

# init model and tokenizer
model = AutoModel.from_pretrained('DLight1551/internlm-xcomposer-vl-7b-qinstruct-full', trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('DLight1551/internlm-xcomposer-vl-7b-qinstruct-full', trust_remote_code=True)
model.tokenizer = tokenizer

image_paths = [
        "../datasets/AGIQA-3K/database/",
        "../datasets/1024x768/",
        "../datasets/SPAQ/",
        "../datasets/FLIVE_Database/database/",
        "../datasets/LIVEC/Images/",
        "../datasets/CGIQA-6K/database/",
        "../datasets/kadid10k/images/",
]

json_prefix = "../datasets/json/"
jsons = [
        json_prefix + "agi.json",
        json_prefix + "koniq.json",
        json_prefix + "spaq.json",
        json_prefix + "flive.json",
        json_prefix + "livec.json",
        json_prefix + "cgi.json",
        json_prefix + "kadid.json",
]

def get_logits(model, text, image_path):
    image = Image.open(image_path).convert("RGB")
    with torch.cuda.amp.autocast():
        image = model.vis_processor(image).unsqueeze(0).to(model.device)
        img_embeds = model.encode_img(image)
    prompt_segs = text.split('<ImageHere>')
    prompt_seg_tokens = [ 
        model.tokenizer(seg,
                             return_tensors='pt',
                             add_special_tokens=i == 0). 
        to(model.internlm_model.model.embed_tokens.weight.device).input_ids
        for i, seg in enumerate(prompt_segs)
    ]   
    prompt_seg_embs = [ 
        model.internlm_model.model.embed_tokens(seg)
        for seg in prompt_seg_tokens
    ]
    prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
        
    prompt_embs = torch.cat(prompt_seg_embs, dim=1)
    
    return model.internlm_model(
        inputs_embeds=prompt_embs).logits[:,-1]

for image_path, input_json in zip(image_paths, jsons):
    with open(input_json) as f:
        iqa_data = json.load(f)
    
    for i, llddata in enumerate(tqdm(iqa_data, desc=image_path)):
        message = "Rate the quality of the image."
        
        llddata["logit_good"] = 0.
        llddata["logit_poor"] = 0.
        
        images = [image_path+llddata["img_path"]]
        for image in images:
            # 1st dialogue turn
            output_logits = get_logits(model, message, image)
            probs, inds = output_logits.sort(dim=-1,descending=True)
            lgood, lpoor = output_logits[0,18682].item(), output_logits[0,5527].item()
            llddata["logit_good"] += lgood
            llddata["logit_poor"] += lpoor

        with open(f"results/mix-internlm_xcomposer_vl/{input_json.split('/')[-1]}", "a") as wf:
            json.dump(llddata, wf)