## Model Zoo


### Huggingface


All weights are converted into Huggingface format and totally compatible with the base repositories ([LLaVA](https://github.com/haotian-liu/LLaVA/), [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl/), [InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer)). See our [quick start](../README.md#quick-start) on how to use them.

_Released_:

- [LLaVA-v1.5-7B (mix)](https://huggingface.co/teowu/llava_v1.5_7b_qinstruct_preview_v0.1), HF-path: `teowu/llava_v1.5_7b_qinstruct_preview_v0.1`
- [LLaVA-v1.5-13B (mix)](https://huggingface.co/teowu/llava_v1.5_13b_qinstruct_preview_v0.1), HF-path: `teowu/llava_v1.5_13b_qinstruct_preview_v0.1`
- [mPLUG-Owl-2 (mix)](https://huggingface.co/teowu/mplug_owl2_7b_448_qinstruct_preview_v0.1), HF-path: `teowu/mplug_owl2_7b_448_qinstruct_preview_v0.1`
- [InternLM-XComposer-VL (mix)](https://huggingface.co/DLight1551/internlm-xcomposer-vl-7b-qinstruct-full), HF-path: `DLight1551/internlm-xcomposer-vl-7b-qinstruct-full`


### ModelScope

If your server is facing poor connection to Huggingface, we provide an alternative way to download weights from ModelScope. Different from direct Huggingface weights, you need to use the two following steps to load them:

#### Step 1: Download Weights


The links are as follows (WIP):


_Released_:

- [LLaVA-v1.5-7B (mix)](https://www.modelscope.cn/models/qfuture/llava_v1.5_7b_qinstruct_preview_v0.1), ModelScope-path: `qfuture/llava_v1.5_7b_qinstruct_preview_v0.1`
- [LLaVA-v1.5-13B (mix)](https://www.modelscope.cn/models/qfuture/llava_v1.5_13b_qinstruct_preview_v0.1), ModelScope-path: `qfuture/llava_v1.5_13b_qinstruct_preview_v0.1`
- [mPLUG-Owl-2 (mix)](https://www.modelscope.cn/models/qfuture/mplug_owl_2_qinstruct_preview_v0.1), ModelScope-path: `qfuture/mplug_owl_2_qinstruct_preview_v0.1`


_Coming Soon_:

- InternLM-XComposer-VL (mix)

To use them, you need to install `Git LFS` and then clone the repositories directly from ModelScope, under the main directory of Q-Instruct.

```shell
git clone https://www.modelscope.cn/models/qfuture/$MODEL_NAME_qinstruct_preview_v0.1.git
```

#### Step 2: Redirect the Model Paths to Your Local Directory

After that, modify the `model_path` in [quick start](../README.md#quick-start) to the local path (i.e. `$MODEL_NAME_qinstruct_preview_v0.1`) to smoothly load the weights downloaded from ModelScope.


See the Example Code for Single Query on LLaVA-v1.5-7B below:
    
```python
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
model_path = "llava_v1.5_7b_qinstruct_preview_v0.1/" ## Modify Here to Your Local Relative Path ##
prompt = "Rate the quality of the image. Think step by step."
image_file = "fig/sausage.jpg"
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
})()
eval_model(args)
```
