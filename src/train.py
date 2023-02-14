import argparse
from peft import LoraConfig, get_peft_model
from util import freeze_model, print_trainable_parameters
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_peft(args):
    # Get output dir from args
    output_dir = args.output_dir if args.output_dir else f"../models/{args.peft_model}"

    # Load our model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
    )

    # Allow use of a custom tokenizer
    tokenizer_name = args.tokenizer_model if args.tokenizer_model else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Check if we need to add a pad token, and if so, add it
    # and resize the model accordingly
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        base_model.resize_token_embeddings(len(tokenizer))

    # Freeze the base model before we add the adapter
    freeze_model(base_model)

    # Load our Lora-adapted model
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=args.lora_task_type,
    )
    model = get_peft_model(base_model, lora_config)
    print_trainable_parameters(model)

    # Load our dataset
    data = load_dataset(
        args.dataset_name, split=args.dataset_split, data_dir=args.dataset_datadir
    )
    train_data = data.map(
        lambda samples: tokenizer(samples[args.dataset_train_col]), batched=True
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Load our trainer!
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=args.fp16,
            logging_steps=args.logging_steps,
            output_dir=output_dir,
        ),
        data_collator=data_collator,
    )

    # Silence cache warnings. This needs to be re-enabled for inference
    model.config.use_cache = False

    # Do the run!
    trainer.train()

    # Save the model to disk
    model.save_pretrained(output_dir)

    # Maybe send it to the hub
    if args.push_to_hub:
        model.push_to_hub(args.peft_model)


def extract_args(args):
    parser = argparse.ArgumentParser()

    # ===
    # Allow either a whole base name, or the user and model id
    # ===
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--peft_model", type=str, required=True)

    # ===
    # Allow a custom tokenizer model
    # ===
    parser.add_argument("--tokenizer_model", type=str,
                        required=False, default=None)

    # ===
    # Require a dataset and associated details
    # ===
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument("--dataset_name", type=str, required=True)
    dataset_group.add_argument(
        "--dataset_split", type=str, required=False, default="train"
    )
    dataset_group.add_argument(
        "--dataset_datadir", type=str, required=False, default=None
    )
    dataset_group.add_argument("--dataset_train_col", type=str, required=True)

    # ===
    # Allow custom hyperparams for trainer and PEFT method of choice
    # ===

    # Trainer params
    trainer_hparams = parser.add_argument_group("Trainer")
    trainer_hparams.add_argument(
        "--batch_size", type=int, required=False, default=4)
    trainer_hparams.add_argument(
        "--gradient_accumulation_steps", type=int, required=False, default=4
    )
    trainer_hparams.add_argument(
        "--warmup_steps", type=int, required=False, default=100
    )
    trainer_hparams.add_argument(
        "--max_steps", type=int, required=False, default=1000)
    trainer_hparams.add_argument(
        "--learning_rate", type=float, required=False, default=1e-4
    )
    trainer_hparams.add_argument(
        "--fp16", type=bool, required=False, default=True)
    trainer_hparams.add_argument(
        "--logging_steps", type=int, required=False, default=100
    )

    # Lora param
    lora_hparams = parser.add_argument_group("Lora")
    lora_hparams.add_argument("--lora_r", type=int, required=False, default=16)
    lora_hparams.add_argument("--lora_alpha", type=int,
                              required=False, default=32)
    lora_hparams.add_argument(
        "--lora_dropout", type=float, required=False, default=0.05
    )
    lora_hparams.add_argument("--lora_bias", type=str,
                              required=False, default="none")
    lora_hparams.add_argument(
        "--lora_task_type", type=str, required=False, default="CAUSAL_LM"
    )

    # ===
    # Require an output directory
    # ===
    parser.add_argument("--output_dir", type=str, required=False)

    # ===
    # Should we push to HuggingFace Hub?
    # ===
    parser.add_argument("--push_to_hub", type=bool,
                        required=False, default=False)

    # Parse the arguments
    return parser.parse_args(args)


if __name__ == "__main__":
    import sys

    args = extract_args(sys.argv[1:])
    train_peft(args)
