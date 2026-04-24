import json
import os
# 可按需修改
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import torch
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, Qwen3ForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument("--input_jsonl", default="test.json", help="Path to input .jsonl or JSON array file")
parser.add_argument("--output_jsonl", default="wrong_samples_vanilla.jsonl", help="Path to save wrong samples")
parser.add_argument("--model_path", default="/home/Datasets/Hf_model/Qwen3-8B/")
parser.add_argument("--deepseek_api_key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
parser.add_argument("--deepseek_base_url", default="https://api.deepseek.com")
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--temperature", type=float, default=0.0)
args = parser.parse_args()

ds_client = OpenAI(api_key=args.deepseek_api_key, base_url=args.deepseek_base_url)

JUDGE_SYSTEM = (
    "You are a strict answer evaluator. "
    "Given a question, a model's predicted answer, and the gold standard answer, "
    "determine whether the predicted answer is semantically equivalent to the gold answer. "
    "Reply ONLY with a JSON object in the format:\n"
    '{"is_correct": true/false, "reason": "<one sentence explanation>"}'
)

def judge_with_deepseek(question: str, prediction: str, gold: str) -> dict:
    user_msg = (
        f"Question: {question}\n"
        f"Predicted answer: {prediction}\n"
        f"Gold answer: {gold}"
    )
    try:
        resp = ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        return {"is_correct": False, "reason": f"Judge API error: {e}"}

def build_prompt(sample: dict) -> str:
    question = sample.get("question", "")
    context_raw = sample.get("context", None)

    if context_raw is None:
        context_str = ""
    elif isinstance(context_raw, str):
        context_str = context_raw
    elif isinstance(context_raw, dict):
        titles = context_raw.get("title", [])
        sentences = context_raw.get("sentences", [])
        parts = []
        for title, sents in zip(titles, sentences):
            parts.append(f"Title: {title}\n" + " ".join(sents))
        context_str = "\n\n".join(parts)
    else:
        context_str = str(context_raw)

    if context_str:
        return (
            "Answer the question based on the context below. "
            "Provide your reasoning before you give the answer.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}"
        )
    return f"Answer the following question.\n\nQuestion: {question}"

def load_samples(path: str):
    with open(path, "r", encoding="utf-8") as fin:
        raw = fin.read().strip()

    if not raw:
        return []

    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("JSON content must be a list when using array format.")
        return data

    samples = []
    for i, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[skip] line {i} JSON decode error: {e}")
    return samples

print("Loading tokenizer and vanilla Qwen3 model...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = Qwen3ForCausalLM.from_pretrained(
    args.model_path,
    attn_implementation="flash_attention_2",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

def generate_answer(prompt: str) -> str:
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=(args.temperature > 0),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(
        generated_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()

print(f"Reading data from: {args.input_jsonl}")
samples = load_samples(args.input_jsonl)
print(f"Loaded {len(samples)} samples")

total = 0
correct = 0
wrong = 0

with open(args.output_jsonl, "w", encoding="utf-8") as fout:
    for sample in tqdm(samples, desc="Evaluating vanilla Qwen3"):
        question = sample.get("question", "")
        gold_answer = sample.get("answer", "")

        prompt = build_prompt(sample)
        prediction = generate_answer(prompt)

        judge = judge_with_deepseek(question, prediction, gold_answer)
        is_correct = bool(judge.get("is_correct", False))
        reason = judge.get("reason", "")

        total += 1
        if is_correct:
            correct += 1
        else:
            wrong += 1
            record = {
                "id": sample.get("id", total),
                "question": question,
                "gold_answer": gold_answer,
                "prediction": prediction,
                "judge_reason": reason,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        tqdm.write(
            f"[{total}] correct={correct} wrong={wrong} | "
            f"{'OK' if is_correct else 'BAD'} | {reason}"
        )

print("\n========== Evaluation Summary ==========")
print(f"Total samples : {total}")
if total > 0:
    print(f"Correct       : {correct}  ({correct / total * 100:.1f}%)")
    print(f"Wrong         : {wrong}  ({wrong / total * 100:.1f}%)")
else:
    print("Correct       : 0")
    print("Wrong         : 0")
print(f"Wrong samples saved to: {args.output_jsonl}")