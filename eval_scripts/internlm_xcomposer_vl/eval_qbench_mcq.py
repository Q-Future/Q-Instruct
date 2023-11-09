from transformers import AutoModel, AutoTokenizer
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM

import random
from copy import deepcopy
import io
import os
import base64
import torch
import numpy as np
from transformers import StoppingCriteria, StoppingCriteriaList
import os
from typing import Optional
import xlsxwriter
import pandas as pd
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import torchvision

torch.set_grad_enabled(False)

torch.manual_seed(1234)

model = AutoModel.from_pretrained('DLight1551/internlm-xcomposer-vl-7b-qinstruct-full', trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(tgt_dir, trust_remote_code=True)
model.tokenizer = tokenizer

def generate_answer(model, text, image_path):
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
    
    outputs = model.internlm_model.generate(
        inputs_embeds=prompt_embs,
        max_new_tokens=5,
        num_beams=5,
        do_sample=False,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1.0,
        stopping_criteria=stopping_criteria,
    )   
    #print (outputs)
    output_token = outputs[0]
    if output_token[0] == 0:
        output_token = output_token[1:]
    if output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token,
                                              add_special_tokens=False)

    output_text = output_text.split(model.eoa)[0]
    output_text = output_text.split('<|Bot|>')[-1].strip()
    return output_text

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

stop_words_ids = [
                  torch.tensor([103027]).cuda(), ### end of human
                  torch.tensor([103028]).cuda(), ### end of bot
                 ]
stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)])

## define split/language here ##
lang = "en" # en | zh
split = "test" # dev | test (not supported for you)
## define split/language here ##

import json
from tqdm import tqdm
if lang == "en":
    with open(f"llvisionqa_{split}.json") as f:
        llvqa_data = json.load(f)
elif lang == "zh":
    zh_split = "验证集" if split == "dev" else "测试集"
    with open(f"质衡-问答-{zh_split}.json") as f:
        llvqa_data = json.load(f)
else:
    raise NotImplementedError("Q-Bench does not support languages other than English (en) and Chinese (zh) yet. Contact us (https://github.com/Q-Future/Q-Bench/) to convert  Q-Bench into more languages.")
    

correct = np.zeros((3,4))
all_ = np.zeros((3,4))
answers = {}
for llddata in tqdm((llvqa_data)):
    t, c = llddata["type"], llddata["concern"]
        
    options_prompt = ''
    for choice, ans in zip(["A.", "B.", "C.", "D."], llddata["candidates"]):
        options_prompt += f"{choice} {ans}\n"
        if "correct_ans" in llddata and ans == llddata["correct_ans"]:
            correct_choice = choice[0]
            
    img_prompt = ' <|User|>:<ImageHere>'
    txt_prompt = 'Please answer this question by choosing the correct choice.'
    context = 'N/A'
    mid_prompt = 'Context: ' + context + '\nQuestion: ' + llddata["question"] + '\nOptions: ' + options_prompt
    ans_prompt = ' <|Bot|>: Answer: The answer is'
    text = img_prompt + txt_prompt + mid_prompt + '<TOKENS_UNUSED_0>' + ans_prompt
    print(text)

    img_path = f"../datasets/LLVQA/images/" + llddata["img_path"]
    # 1st dialogue turn
    response = generate_answer(model, text, img_path)
    all_[t][c] += 1
    if response[0] not in ['A', 'B', 'C', 'D']:
        print("[Response]: {}, [Correct Ans]: {}".format(response, correct_choice))
    if split == 'dev' and response[0] == correct_choice:
        correct[t][c] += 1
        
print (correct.sum(1)/all_.sum(1))
print (correct.sum(0)/all_.sum(0))
print ("Final Correctness": correct.sum()/all_.sum())