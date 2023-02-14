# PEFT Trainer Script
A simple Python script designed to train a PEFT adapter using Lora.<br>
`Version 0.0.1`

## Requirements
Use an appropriate GPU for your workload. This script does not do int8 training, so assume full model weights.


>A basic rule of thumb is to assume you'll typically around require 8GB VRAM per billion params in the model at training time, and 2GB VRAM per billion params at inference-time for the same model.

## Basic Usage
Here is a minimal example to finetune gpt2 on famous English quotes:
```bash
# Install Python requirements
pip install -r requirements.txt

# Minimal example, will output Lora model to models/calum/my-custom-gpt2
cd src && python train.py \
--base_model "gpt2" \
--peft_model "calum/my-custom-gpt2" \
--dataset_name "Abirate/english_quotes" \
--dataset_train_col "quote" \
--max_steps 1000
```

## Advanced Usage
Of course, you can customize use of the script to your liking. Here is a full list of valid arguments to the training script, separated by their use case.


### Model Tokenizer names:
| Argument | Description | Required | Default |
| --- | --- | --- | --- |
| `base_model` | The base model to use. | `Yes` | `None` |
| `peft_model` | The name of your model, in huggingface url format. | `Yes` | `None` |
| `tokenizer_model` | The tokenizer model to use. | `No` | `None` |

### Dataset:
| Argument | Description | Required | Default |
| --- | --- | --- | --- |
| `dataset_name` | The name of the dataset to use. | `Yes` | `None` |
| `dataset_split` | The name of the dataset split to use (eg train, validation, test). | `No` | `"train"` |
| `dataset_data_dir` | The name of the dataset data directory to use. | `No` | `None` |
| `dataset_train_col` | The name of the column to use for training. | `Yes` | `None` |

### Training:
| Argument | Description | Required | Default |
| --- | --- | --- | --- |
| `batch_size` | The batch size to use for training. | `No` | `4` |
| `gradient_accumulation_steps` | Integer by which to divide a given batch of data into smaller mini batche sizes to reduce memory usage during training. | `No` | `4` |
| `warmup_steps` | The number of warmup steps to use for training, during which the learning rate will be higher. | `No` | `100` |
| `max_steps` | The maximum number of steps to train for. | `No` | `1000` |
| `learning_rate` | The learning rate to use during training. | `No` | `1e-4` |
| `fp16` | Whether to use mixed precision training. | `No` | `True` |
| `logging_steps` | The interval of steps after which we log training progress. | `No` | `100` |

### Lora:
| Argument | Description | Required | Default |
| --- | --- | --- | --- |
| `lora_r` | The name of the Lora model to use. | `No` | `16` |
| `lora_alpha` | The name of the Lora model to use. | `No` | `32` |
| `lora_dropout` | The name of the Lora model to use. | `No` | `0.05` |
| `lora_bias` | The name of the Lora model to use. | `No` | `None` |
| `lora_task_type` | The name of the Lora model to use. | `No` | `"CAUSAL_LM"` |

### Misc:
| Argument | Description | Required | Default |
| --- | --- | --- | --- |
| `output_dir` | The output directory to use for saving the model. | `No` | `../models/{peft_model}` |
| `push_to_hub` | Whether to push the model to huggingface hub. | `No` | `False` |