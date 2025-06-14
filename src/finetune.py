import os
os.environ["XFORMERS_DISABLE_MEM_EFF_ATTENTION"] = "1"
os.environ["XFORMERS_FORCE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def add_suffix(sample):
    """
    텍스트에 레이블(positive/negative)을 접미사로 추가합니다.
    """
    if sample['label'] == 0:
        label = 'negative'
    else:
        label = 'positive'
    sample['text'] = sample['text'] + ' // ' + label
    return sample

def remove_commawhitespace(sample):
    """
    쉼표 앞의 공백을 제거합니다.
    """
    sample['text'] = sample['text'].replace(" ,",",")
    return sample

def remove_qwhitespace(sample):
    """
    물음표 주변의 공백을 제거합니다.
    """
    sample['text'] = sample['text'].replace(" ? ","?")
    return sample

def remove_dwhitespace(sample):
    """
    마침표 주변의 공백을 제거합니다.
    """
    sample['text'] = sample['text'].replace(" . ",".")
    return sample

def remove_exwhitespace(sample):
    """
    느낌표 주변의 공백을 제거합니다.
    """
    sample['text'] = sample['text'].replace(" ! ","!")
    return sample

def remove_swhitespace(sample):
    """
    's 앞의 공백을 제거합니다.
    """
    sample['text'] = sample['text'].replace(" 's","'s")
    return sample

def strip_whitespace(sample):
    """
    텍스트 끝의 공백을 제거합니다.
    """
    sample['text'] = sample['text'].rstrip()
    return sample

def preprocess_yelp_polarity(dataset_size):
    """
    Yelp Polarity 데이터셋을 전처리하고 학습/테스트로 분할합니다.
    """
    print(f"Loading Yelp Polarity dataset with {dataset_size} samples...")
    dataset = load_dataset("yelp_polarity", split="train", cache_dir="./hf_cache")
    dataset = dataset.map(lambda example: {'len': len(example['text'])})
    med = np.median(np.array(dataset['len']))
    indices = np.where(dataset['len'] < med)[0]
    dataset = dataset.select(indices)
    dataset = dataset.map(remove_commawhitespace)
    dataset = dataset.map(remove_swhitespace)
    dataset = dataset.map(strip_whitespace)
    dataset = dataset.map(add_suffix)
    n = len(dataset['text'])
    all_indices = range(n)
    indices_subset = np.random.choice(all_indices, dataset_size, replace=False)
    dataset = dataset.select(indices_subset)
    train_size = int(0.9 * dataset_size)
    train_indices = np.arange(train_size)
    test_indices = np.arange(train_size, dataset_size)
    dataset_train = dataset.select(train_indices)
    dataset_test = dataset.select(test_indices)
    return DatasetDict({"train": dataset_train, "test": dataset_test})

def preprocess_custom_dataset(output_dir):
    """
    커스텀 CSV 데이터셋을 전처리하고 학습/테스트로 분할합니다.
    """
    csv_path = os.path.join(output_dir + "custom_dataset.csv")
    df = pd.read_csv(csv_path)

    def apply_mappings(row):
        row = remove_commawhitespace(row)
        row = remove_swhitespace(row)
        row = strip_whitespace(row)
        row = add_suffix(row)
        row['len'] = len(row['text'])
        return row

    processed = df.apply(apply_mappings, axis=1)
    processed_df = pd.DataFrame(processed)

    train_size = int(0.9 * len(processed_df))
    train_df = processed_df.iloc[:train_size]
    test_df = processed_df.iloc[train_size:]

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    return dataset

def finetune(args):
    """
    주어진 데이터셋으로 LLaMA 모델을 파인튜닝합니다.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.type == "paper":
        dataset = preprocess_yelp_polarity(args.dataset_size)
    elif args.type == "project":
        dataset = preprocess_custom_dataset(args.output_dir)

    model_name = "unsloth/Llama-3.2-3B-Instruct"
    print("Loading Llama-3.2-3B-Instruct in 4bit with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        load_in_4bit = True,
        dtype = torch.float16,
        device_map = "auto",
        cache_dir="./hf_cache",
    )

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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("Cleared CUDA cache and collected IPC before training.")

    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_datasets.set_format(type="torch")

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    training_args = TrainingArguments(
        output_dir=f"finetuned_llama_{args.type}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        fp16=args.fp16,
    )

    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    print(f"Saving model to finetuned_llama_{args.type} ...")
    model.save_pretrained(f"finetuned_llama_{args.type}")
    tokenizer.save_pretrained(f"finetuned_llama_{args.type}")
    print("Finetuning complete.")
    return f"finetuned_llama_{args.type}"
