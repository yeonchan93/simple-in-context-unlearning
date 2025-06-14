import os
import json
import random
import pandas as pd
from tqdm.auto import tqdm
from openai import OpenAI

import names
from dotenv import load_dotenv

load_dotenv()

response_schema = {
    "name": "medical_record_response",
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "gender": {"type": "string"},
            "diagnosis": {"type": "string"},
            "diabetes": {"type": "integer"},
            "blood_pressure": {"type": "string"},
            "cholesterol": {"type": "integer"},
            "outcome": {"type": "integer"}
        },
        "required": [
            "name", "age", "gender", "diagnosis", "diabetes", "blood_pressure", "cholesterol", "outcome"
        ],
        "additionalProperties": False
    },
    "strict": True
}
batch_output_file = "batch_output.jsonl"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
api_endpoint = "/v1/chat/completions"

def generate_random_names(args):
    """
    무작위로 이름과 성별을 생성합니다.
    """
    n = args.dataset_size
    names_list = []
    for _ in range(n):
        gender = random.choice(['male', 'female'])
        if gender == 'male':
            name = names.get_full_name(gender='male')
        else:
            name = names.get_full_name(gender='female')
        names_list.append({"name": name, "gender": gender.capitalize()})
    return names_list

def create_openai_batch_jsonl(args, names_list):
    """
    OpenAI 배치 처리를 위한 입력 JSONL 파일을 생성합니다.
    """
    jsonl_path = os.path.join(args.output_dir, "batch_input.jsonl")
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    system_prompt = (
        "You are a medical assistant. Given a patient name, generate a realistic but synthetic medical record with the following fields: "
        "age (18-90), gender (Male/Female), diagnosis (a plausible medical diagnosis), diabetes (1 if diabetic, 0 otherwise), "
        "blood_pressure (e.g., 120/80), cholesterol (mg/dL), outcome (1 if patient has a disease, 0 otherwise)."
    )
    model = os.getenv("OPENAI_MODEL")
    n = len(names_list)
    half = n // 2
    for idx, entry in tqdm(enumerate(names_list), total=n, desc="Preparing batch input"):
        if idx < half:
            outcome_instruction = "Set outcome to 1 (patient has a disease)."
        else:
            outcome_instruction = "Set outcome to 0 (patient does not have a disease)."
        user_prompt = (
            f"Patient name: {entry['name']}\n"
            f"Gender: {entry['gender']}\n"
            "Generate the following fields for this patient:\n"
            "- age (18-90)\n"
            "- diagnosis (a plausible medical diagnosis)\n"
            "- diabetes (1 if diabetic, 0 otherwise)\n"
            "- blood_pressure (e.g., 120/80)\n"
            "- cholesterol (mg/dL)\n"
            "- outcome (1 if patient has a disease, 0 otherwise)\n"
            f"{outcome_instruction}\n"
            "Respond in JSON format according to the provided schema."
        )
        record = {
            "custom_id": str(idx),
            "method": "POST",
            "url": api_endpoint,
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": response_schema
                }
            }
        }
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return jsonl_path

def parse_openai_batch_output(args, batch_output_paths):
    """
    OpenAI 배치 출력 파일(.jsonl)을 파싱하여 하나의 CSV 파일로 만듭니다.
    """
    csv_path = os.path.join(args.output_dir, "custom_dataset.csv")
    records = []
    if isinstance(batch_output_paths, str):
        batch_output_paths = [batch_output_paths]
    for batch_output_path in batch_output_paths:
        with open(batch_output_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                content_str = obj["response"]["body"]["choices"][0]["message"]["content"]
                content = json.loads(content_str)
                label = content.get("outcome")
                text_fields = {k: v for k, v in content.items() if k != "outcome"}
                text = ", ".join(f"{k}: {v}" for k, v in text_fields.items())
                records.append({"text": text, "label": label})
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved medical dataset to {csv_path}")
    return csv_path

def main_dataset_task(args):
    """
    데이터셋 생성 및 파싱을 위한 메인 함수입니다.
    """
    batch_output_files = [
        os.path.join(args.output_dir, f"batch_output_{i}.jsonl") for i in range(1, 4)
    ]
    all_exist = all(os.path.exists(f) for f in batch_output_files)
    single_output_path = os.path.join(args.output_dir, "batch_output.jsonl")
    if not (all_exist or os.path.exists(single_output_path)):
        print(f"[INFO] No batch output files found. Creating batch_input.jsonl for OpenAI batch processing.")
        names_list = generate_random_names(args)
        batch_input_path = create_openai_batch_jsonl(args, names_list)
        file_response = client.files.create(
            file=open(batch_input_path, "rb"),
            purpose="batch"
        )
        batch_response = client.batches.create(
            input_file_id=file_response.id,
            endpoint=api_endpoint,
            completion_window="24h"
        )
    
        print(f"[INFO] Created batch input at {batch_input_path}. Please run the OpenAI batch job and place the output at {single_output_path} or as batch_output_1.jsonl, batch_output_2.jsonl, batch_output_3.jsonl.")
    else:
        if all_exist:
            print(f"[INFO] Found all batch output files. Parsing and combining to custom_dataset.csv ...")
            csv_path = parse_openai_batch_output(args, batch_output_files)
        else:
            print(f"[INFO] Found {single_output_path}. Parsing to custom_dataset.csv ...")
            csv_path = parse_openai_batch_output(args, single_output_path)
        print(f"[INFO] Dataset saved to {csv_path}")
