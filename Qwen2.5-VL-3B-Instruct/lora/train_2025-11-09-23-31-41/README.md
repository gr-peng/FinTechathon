---
library_name: peft
license: other
base_model: Qwen/Qwen2.5-VL-3B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: train_2025-11-09-23-31-41
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_2025-11-09-23-31-41

This model is a fine-tuned version of [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) on the [Easy Dataset] [awsUfO7kIcbJ] ShareGPT dataset.
It achieves the following results on the evaluation set:
- Loss: 1.0875
- Num Input Tokens Seen: 4061024

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- total_eval_batch_size: 4
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.15.2
- Transformers 4.51.3
- Pytorch 2.8.0+cu128
- Datasets 3.6.0
- Tokenizers 0.21.1