import os
os.environ["XFORMERS_DISABLE_MEM_EFF_ATTENTION"] = "1"
os.environ["XFORMERS_FORCE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import argparse
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, DatasetDict
from unsloth import FastLanguageModel
from transformers import BitsAndBytesConfig

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

def remove_swhitespace(sample):
    sample['text'] = sample['text'].replace(" 's","'s")
    return sample

def strip_whitespace(sample):
    sample['text'] = sample['text'].rstrip()
    return sample

def preprocess_yelp_polarity(dataset_size, seed=42):
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
    np.random.seed(seed)
    indices_subset = np.random.choice(all_indices, dataset_size, replace=False)
    dataset = dataset.select(indices_subset)
    # Split into train/test (90/10 split)
    train_size = int(0.9 * dataset_size)
    train_indices = np.arange(train_size)
    test_indices = np.arange(train_size, dataset_size)
    dataset_train = dataset.select(train_indices)
    dataset_test = dataset.select(test_indices)
    return DatasetDict({"train": dataset_train, "test": dataset_test})

def construct_icl_context(example, context_examples, n_ctxt, label_flipping_method="first-k"):
    """
    Construct ICL context for a single example, with n_ctxt-1 context_examples.
    The first label is flipped if label_flipping_method == "first-k".
    """
    # Parse text and label from main example
    main_text, main_label = example["text"], example["label"]
    # Remove the label suffix for context construction
    main_text_base = main_text.rsplit("//", 1)[0].strip()
    # Prepare context
    context_texts = []
    # First: flip label for the main example
    if main_label == 1:
        flipped_label = "negative" if label_flipping_method == "first-k" else "positive"
    else:
        flipped_label = "positive" if label_flipping_method == "first-k" else "negative"
    context_texts.append(f"{main_text_base} // {flipped_label}")
    # Add n_ctxt-1 context examples with their true labels
    for ctx in context_examples:
        ctx_text, ctx_label = ctx["text"], ctx["label"]
        ctx_text_base = ctx_text.rsplit("//", 1)[0].strip()
        ctx_label_str = "positive" if ctx_label == 1 else "negative"
        context_texts.append(f"{ctx_text_base} // {ctx_label_str}")
    # The query is the original main_text (with true label)
    query = main_text
    # Join context and query
    icl_prompt = "\n".join(context_texts) + "\n" + query
    return icl_prompt

def get_stable_logit_loss(out, label, id_pos, id_neg, eps=1e-35):
    """
    Compute stable logit loss for binary classification.
    """
    if label == 0:
        Y = id_neg
    else:
        Y = id_pos
    # out['scores'][0][-1] is the logits for the next token
    probs = torch.nn.functional.softmax(out['scores'][0][-1], dim=0)
    class_log_prob = torch.log(probs[Y] + eps)
    complement_class_sum = torch.sum(probs) - probs[Y]
    score = class_log_prob - torch.log(complement_class_sum + eps)
    return score.item(), class_log_prob.item(), torch.log(complement_class_sum + eps).item()

def compute_token_pred(out, tokenizer):
    pred_token = tokenizer.decode(out['sequences'][0][-1]).strip()
    return pred_token

def unlearn(model_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_csv', type=str, default="./assets/icul_results.csv", help="CSV file to save results")
    parser.add_argument('--dataset_size', type=int, default=25000)
    parser.add_argument('--n_ctxt', type=int, default=2, help="Number of context examples (including the forget point)")
    parser.add_argument('--label_flipping_method', type=str, default="first-k", choices=["first-k", "last-k"])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load and preprocess Yelp Polarity dataset
    dataset = preprocess_yelp_polarity(args.dataset_size, seed=args.seed)
    test_dataset = dataset["test"]
    train_dataset = dataset["train"]

    # 2. Load Unsloth Llama-3.2-3B-Instruct in 4bit
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_dir,
        load_in_4bit = True,
        dtype = torch.float16,
        device_map = "auto",
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Prepare for evaluation
    results = {
        "idx": [],
        "icl_prompt": [],
        "true_label": [],
        "flipped_label": [],
        "pred_token": [],
        "logit_loss": [],
        "class_log_prob": [],
        "complement_log_prob": [],
    }

    # Get token ids for " positive" and " negative"
    id_pos = tokenizer(" positive", return_tensors="pt")["input_ids"][0][0].item()
    id_neg = tokenizer(" negative", return_tensors="pt")["input_ids"][0][0].item()

    print("Evaluating in-context unlearning...")
    for idx in range(len(test_dataset)):
        example = test_dataset[idx]
        # Sample n_ctxt-1 context examples from train set (random, without replacement)
        context_indices = np.random.choice(len(train_dataset), args.n_ctxt-1, replace=False)
        context_examples = [train_dataset[int(i)] for i in context_indices]
        icl_prompt = construct_icl_context(example, context_examples, args.n_ctxt, args.label_flipping_method)
        # Tokenize
        tokenized = tokenizer(icl_prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=args.max_length)
        input_ids = tokenized["input_ids"].cuda() if torch.cuda.is_available() else tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"].cuda() if torch.cuda.is_available() else tokenized["attention_mask"]
        # Run model
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1]+1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        # Compute metrics
        logit_loss, class_log_prob, complement_log_prob = get_stable_logit_loss(out, example["label"], id_pos, id_neg)
        pred_token = compute_token_pred(out, tokenizer)
        # Store results
        results["idx"].append(idx)
        results["icl_prompt"].append(icl_prompt)
        results["true_label"].append("positive" if example["label"] == 1 else "negative")
        results["flipped_label"].append("negative" if example["label"] == 1 else "positive")
        results["pred_token"].append(pred_token)
        results["logit_loss"].append(logit_loss)
        results["class_log_prob"].append(class_log_prob)
        results["complement_log_prob"].append(complement_log_prob)
        if (idx+1) % 100 == 0:
            print(f"Processed {idx+1}/{len(test_dataset)} samples...")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
