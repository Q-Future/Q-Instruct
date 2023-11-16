import numpy as np
from scipy.stats import spearmanr, pearsonr
import json, glob, numpy as np

def wa_any(score_dict, pos_keys, mid_keys, neg_keys):
    
    pos_logits = []
    mid_logits = []
    neg_logits = []
    
    for pkey in pos_keys:
        pos_logits += [score_dict[pkey]]

    for mkey in mid_keys:
        mid_logits += [score_dict[mkey]]
        
    if not mid_logits:
        mid_logits = [0]
        
    for nkey in neg_keys:
        neg_logits += [score_dict[nkey]]
    return wa(np.mean(pos_logits), np.mean(mid_logits), np.mean(neg_logits))

def wa(a, b, c, t=1):
    a, b, c = a/t, b/t, c/t
    if b == 0:
        return np.exp(a/100) / (np.exp(a/100) + np.exp(c/100))
    expa = np.exp(a) / (np.exp(a) + np.exp(b) + np.exp(c))
    expb = np.exp(b) / (np.exp(a) + np.exp(b) + np.exp(c))
    return expa + 0.5*expb

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", type=str, default="results/mix-llava-v1.5-7b-boost/livevqc.json")
    
    args = parser.parse_args()
    with open(args.json_file) as f:
        s = f.read().replace("}{", "},{")
        if s[0] != "[":
            s = "[" + s + "]"
        d = json.loads(s)

    if "konvid" in args.json_file or "livevqc" in args.json_file:
        print("Using the boost setting for videos")
        # this will work better for videos, not knowing why
        pos_list, mid_list, neg_list = ["good", "high", "fine"], ["average", "medium", "acceptable"], ["poor", "low", "bad"]
    else:
        print("Using the boost setting for images")
        # this will work better for images
        pos_list, mid_list, neg_list = ["good"], ["average"], ["poor"]
    
    try:
        gt_scores = [float(di["gt_score"]) for di in d]
    except:
        # compatible with v0
        gt_scores = [float(di["gt"]) for di in d]
        
    pr_scores = [wa_any(di["logits"],pos_list, mid_list, neg_list) for di in d]
    
    if "cgi" in args.json_file:
        s = spearmanr(gt_scores[:3000], pr_scores[:3000])[0]
        p = pearsonr(gt_scores[:3000], pr_scores[:3000])[0]
        s += spearmanr(gt_scores[3000:], pr_scores[3000:])[0]
        p += pearsonr(gt_scores[3000:], pr_scores[3000:])[0]
        
        print("Including the *medium-level* token: srcc: {}, prcc: {}".format(s/2,p/2))
        
    else:
        s = spearmanr(gt_scores, pr_scores)[0]
        p = pearsonr(gt_scores, pr_scores)[0]
        print("Including the *medium-level* token: srcc: {}, prcc: {}".format(s,p))
    
    pr_scores = [wa_any(di["logits"],pos_list, [], neg_list) for di in d]

    if "cgi" in args.json_file:
        s = spearmanr(gt_scores[:3000], pr_scores[:3000])[0]
        p = pearsonr(gt_scores[:3000], pr_scores[:3000])[0]
        s += spearmanr(gt_scores[3000:], pr_scores[3000:])[0]
        p += pearsonr(gt_scores[3000:], pr_scores[3000:])[0]
        
        print("Including the *medium-level* token: srcc: {}, prcc: {}".format(s/2,p/2))
        
    else:
        s = spearmanr(gt_scores, pr_scores)[0]
        p = pearsonr(gt_scores, pr_scores)[0]
        print("Including the *medium-level* token: srcc: {}, prcc: {}".format(s,p))