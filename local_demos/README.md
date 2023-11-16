# Demos on Your Local Machine


To run the demos exactly the same as the HF spaces, you will need to slightly modify your environment:


## mPLUG-Owl2

### Installation

Install mPLUG-Owl2 model. If you have already installed it, please directly jump to next step.

```
git clone https://github.com/X-PLUG/mPLUG-Owl.git
cd mPLUG_Owl/mPLUG_Owl2/ 
pip install -e .
```

After normal installation, you need to modify the Gradio to the latest version to run the demo seamlessly.

```shell
pip install gradio==4.1.1
```

### Run Demo

```shell
cd local_demos/mplug_owl_2
gradio app.py
```

## LLaVA-v1.5

To be added soon.

## InternLM-XComposer-VL

To be added soon.