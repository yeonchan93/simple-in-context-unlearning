import os
os.environ["XFORMERS_DISABLE_MEM_EFF_ATTENTION"] = "1"
os.environ["XFORMERS_FORCE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from datasets import load_dataset, DatasetDict
from unsloth import FastLanguageModel
from transformers import BitsAndBytesConfig
from tqdm.auto import tqdm

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

def construct_icl_context(example, context_examples, label_flipping_method="first-k"):
    """
    주어진 예시와 컨텍스트 예시들을 이용해 ICUL 프롬프트를 생성합니다.
    """
    main_text, main_label = example["text"], example["label"]
    main_text_base = main_text.rsplit("//", 1)[0].strip()
    context_texts = []
    if main_label == 1:
        flipped_label = "negative" if label_flipping_method == "first-k" else "positive"
    else:
        flipped_label = "positive" if label_flipping_method == "first-k" else "negative"
    context_texts.append(f"{main_text_base} // {flipped_label}")
    for ctx in context_examples:
        ctx_text, ctx_label = ctx["text"], ctx["label"]
        ctx_text_base = ctx_text.rsplit("//", 1)[0].strip()
        ctx_label_str = "positive" if ctx_label == 1 else "negative"
        context_texts.append(f"{ctx_text_base} // {ctx_label_str}")
    icul_prompt = "\n".join(context_texts) + "\n" + main_text_base + " // " + ("positive" if main_label == 1 else "negative")
    return icul_prompt

def get_stable_logit_loss(out, label, id_pos, id_neg, eps=1e-35):
    """
    이진 분류를 위한 안정적인 logit loss를 계산합니다.
    """
    if label == 0:
        Y = id_neg
    else:
        Y = id_pos
    probs = torch.nn.functional.softmax(out['scores'][0][-1], dim=0)
    class_log_prob = torch.log(probs[Y] + eps)
    complement_class_sum = torch.sum(probs) - probs[Y]
    score = class_log_prob - torch.log(complement_class_sum + eps)
    return score.item(), class_log_prob.item(), torch.log(complement_class_sum + eps).item()

def compute_token_pred(out, tokenizer):
    """
    모델의 출력에서 예측 토큰을 디코딩하여 반환합니다.
    """
    pred_token = tokenizer.decode(out['sequences'][0][-1]).strip()
    return pred_token

def analyze_icul_roc(args):
    """
    ICUL 결과로부터 ROC 곡선을 그리고 AUC를 계산합니다.
    """
    df = pd.read_csv(f"{args.output_dir}icul_results_{args.type}_n{args.n_ctxt-1}.csv")
    y_true = (df["true_label"] == "positive").astype(int).values
    log_p_pos = np.where(
        df["true_label"] == "positive",
        df["class_log_prob"],
        df["complement_log_prob"]
    )
    log_p_neg = np.where(
        df["true_label"] == "positive",
        df["complement_log_prob"],
        df["class_log_prob"]
    )
    
    scores = log_p_pos - log_p_neg
    
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 5))
    sns.set(style="whitegrid")
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ICUL ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Membership Inference ROC Curve (ICUL Results)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}icul_roc_{args.type}_n{args.n_ctxt-1}.png")
    plt.close()
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Result chart saved to {args.output_dir}icul_roc_{args.type}_n{args.n_ctxt-1}.png")

def analyze_icul_accuracy(args):
    """
    ICUL 결과로부터 정확도를 계산하고 시각화합니다.
    """
    df = pd.read_csv(f"{args.output_dir}icul_results_{args.type}_n{args.n_ctxt-1}.csv")
    correct = (df["true_label"] == df["pred_token"]).sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0.0
    print(f"ICUL Accuracy: {accuracy:.4f} ({correct} out of {total})")
    
    baseline = 0.89
    icul = accuracy
    labels = ["Baseline", "ICUL"]
    values = [baseline * 100, icul * 100]
    
    plt.figure(figsize=(5, 5))
    sns.set(style="whitegrid")
    bars = plt.bar(labels, values, color=["gray", "royalblue"])
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("ICUL vs Baseline Accuracy")
    plt.bar_label(bars, fmt="%.1f%%")
    plt.tight_layout()
    out_path = f"{args.output_dir}icul_accuracy_{args.type}_n{args.n_ctxt-1}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Accuracy chart saved to {out_path}")

def unlearn(args, model_dir):
    """
    주어진 데이터셋과 모델로 인컨텍스트 언러닝(ICUL) 평가를 수행합니다.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.type == "paper":
        dataset = preprocess_yelp_polarity(args.dataset_size)
    elif args.type == "project":
        dataset = preprocess_custom_dataset(args.dataset_size)
    test_dataset = dataset["test"]
    train_dataset = dataset["train"]

    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_dir,
        load_in_4bit = True,
        dtype = torch.float16,
        device_map = "auto",
        cache_dir="./hf_cache",
    )
    tokenizer.pad_token = tokenizer.eos_token

    results = {
        "idx": [],
        "icul_prompt": [],
        "true_label": [],
        "flipped_label": [],
        "pred_token": [],
        "logit_loss": [],
        "class_log_prob": [],
        "complement_log_prob": [],
    }

    id_pos = tokenizer(" positive", return_tensors="pt")["input_ids"][0][0].item()
    id_neg = tokenizer(" negative", return_tensors="pt")["input_ids"][0][0].item()

    print("Evaluating in-context unlearning...")
    for idx in tqdm(range(len(test_dataset))):
        example = test_dataset[idx]
        context_indices = np.random.choice(len(train_dataset), args.n_ctxt-1, replace=False)
        context_examples = [train_dataset[int(i)] for i in context_indices]
        icul_prompt = construct_icl_context(example, context_examples, args.label_flipping_method)
        tokenized = tokenizer(icul_prompt, return_tensors="pt", truncation=True, padding="do_not_pad", max_length=args.max_length)
        input_ids = tokenized["input_ids"][0][:-1]
        attention_mask = tokenized["attention_mask"][0][:-1]
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + 1,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
            )
        logit_loss, class_log_prob, complement_log_prob = get_stable_logit_loss(out, example["label"], id_pos, id_neg)
        pred_token = compute_token_pred(out, tokenizer)
        results["idx"].append(idx)
        results["icul_prompt"].append(icul_prompt)
        results["true_label"].append("positive" if example["label"] == 1 else "negative")
        results["flipped_label"].append("negative" if example["label"] == 1 else "positive")
        results["pred_token"].append(pred_token)
        results["logit_loss"].append(logit_loss)
        results["class_log_prob"].append(class_log_prob)
        results["complement_log_prob"].append(complement_log_prob)

    df = pd.DataFrame(results)
    df.to_csv(f"{args.output_dir}icul_results_{args.type}_n{args.n_ctxt-1}.csv", index=False)
    print(f"Results saved to {args.output_dir}icul_results_{args.type}_n{args.n_ctxt-1}.csv")
    analyze_icul_roc(args)
    analyze_icul_accuracy(args)
