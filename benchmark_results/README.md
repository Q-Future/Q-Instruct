## Results

### Answering Multi-Choice Questions

#### Quantitative Results


- `dev` subset of [LLVisionQA](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench).

  <div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="mcq_dev.png">
  </div> 
  
- `test` subset of [LLVisionQA](https://huggingface.co/datasets/nanyangtu/LLVisionQA-QBench).

  <div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="mcq_test.png">
  </div> 

### General Description on Low-level Visual Aspects

#### Quantitative Results

- Results on [LLDescribe](https://huggingface.co/datasets/nanyangtu/LLDescribe-QBench).

  <div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="description.png">
  </div> 


### Image and Video Quality Assessment

#### Quantitative Results

- Impressive results on 8 IQA/VQA datasets, including 3 *never seen* datasets (CGIQA-6K, KADID-10K, KoNViD-10k).

  <div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="iqa_vqa.png">
  </div> 
  
- The results are obtained with text-only instruction tuning, **without any numerical supervision**.