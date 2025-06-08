import os
os.environ["XFORMERS_DISABLE_MEM_EFF_ATTENTION"] = "1"
os.environ["XFORMERS_FORCE_DISABLE"] = "1"
import argparse
import json
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict
import torch

# Unsloth imports
from unsloth import FastLanguageModel
# from unsloth.data import SFTDataset, get_yielding_dataset
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def add_suffix(sample):
    if sample['label'] == 0:
        label = 'negative'
    else:
        label = 'positive'
    sample['text'] = sample['text'] + ' // ' + label
    return sample

def remove_commawhitespace(sample):
    sample['text'] = sample['text'].replace(" ,",",")
    return sample

def remove_qwhitespace(sample):
    sample['text'] = sample['text'].replace(" ? ","?")
    return sample

def remove_dwhitespace(sample):
    sample['text'] = sample['text'].replace(" . ",".")
    return sample

def remove_exwhitespace(sample):
    sample['text'] = sample['text'].replace(" ! ","!")
    return sample

def remove_swhitespace(sample):
    sample['text'] = sample['text'].replace(" 's","'s")
    return sample

def strip_whitespace(sample):
    sample['text'] = sample['text'].rstrip()
    return sample

def preprocess_yelp_polarity(dataset_size):
    print(f"Loading Yelp Polarity dataset with {dataset_size} samples...")
    dataset = load_dataset("yelp_polarity", split="train")
    # Keep documents shorter than the median length
    dataset = dataset.map(lambda example: {'len': len(example['text'])})
    med = np.median(np.array(dataset['len']))
    indices = np.where(dataset['len'] < med)[0]
    dataset = dataset.select(indices)
    # Remove whitespace and add label suffix
    dataset = dataset.map(remove_commawhitespace)
    dataset = dataset.map(remove_swhitespace)
    dataset = dataset.map(strip_whitespace)
    dataset = dataset.map(add_suffix)
    # Select random subset
    n = len(dataset['text'])
    all_indices = range(n)
    indices_subset = np.random.choice(all_indices, dataset_size, replace=False)
    dataset = dataset.select(indices_subset)
    # Split into train/test (90/10 split)
    train_size = int(0.9 * dataset_size)
    train_indices = np.arange(train_size)
    test_indices = np.arange(train_size, dataset_size)
    dataset_train = dataset.select(train_indices)
    dataset_test = dataset.select(test_indices)
    return DatasetDict({"train": dataset_train, "test": dataset_test})

def finetune():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="finetuned_llama3_yelp")
    parser.add_argument('--dataset_size', type=int, default=25000)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load and preprocess Yelp Polarity dataset
    dataset = preprocess_yelp_polarity(args.dataset_size)

    # 2. Load Unsloth Llama-3.1-8B in 4bit
    model_name = "unsloth/Llama-3.2-3B-Instruct"
    print("Loading Llama-3.2-3B-Instruct in 4bit with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        load_in_4bit = True,
        dtype = torch.float16,
        device_map = "auto",
        # use_unsloth = True,
    )

    # Add LoRA adapters for PEFT fine-tuning
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    tokenizer.pad_token = tokenizer.eos_token

    # Clear CUDA cache and collect IPC to free up memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("Cleared CUDA cache and collected IPC before training.")

    # 3. Tokenize dataset
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        # For causal LM, labels are a copy of input_ids
        result["labels"] = result["input_ids"].copy()
        return result

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_datasets.set_format(type="torch")

    # 4. Prepare SFTDataset for Unsloth
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        fp16=args.fp16,
    )

    # 6. Trainer
    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 7. Save model
    print(f"Saving model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Finetuning complete.")
    return args.output_dir
