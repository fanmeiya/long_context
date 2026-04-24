import json
import os
import argparse
import torch
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, Qwen3ForCausalLM

# ─────────────────────────────────────────
# Args
# ─────────────────────────────────────────
parser = argparse.ArgumentParser(description="使用原始 Qwen3 模型重新评估特定样本。")
parser.add_argument("--input_jsonl", default="comparison_results/only_wrong_in_wrong_samples.jsonl", help="要重新评估的样本文件路径。")
parser.add_argument("--output_jsonl", default="re_evaluation_results.jsonl", help="保存新评估结果的路径。")
parser.add_argument("--model_path", default="/home/Datasets/Hf_model/Qwen3-8B/", help="预训练模型路径。")
parser.add_argument("--deepseek_api_key", default=os.environ.get("DEEPSEEK_API_KEY", ""), help="DeepSeek API Key。")
parser.add_argument("--deepseek_base_url", default="https://api.deepseek.com", help="DeepSeek API base URL。")
parser.add_argument("--max_new_tokens", type=int, default=512, help="生成新 token 的最大数量。")
parser.add_argument("--temperature", type=float, default=0.0, help="生成温度，0 表示确定性生成。")
parser.add_argument("--gpu_device", type=str, default="3", help="要使用的 GPU 设备 ID。")
args = parser.parse_args()

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

# ─────────────────────────────────────────
# DeepSeek Judge Client
# ─────────────────────────────────────────
ds_client = OpenAI(api_key=args.deepseek_api_key, base_url=args.deepseek_base_url)

JUDGE_SYSTEM = (
    "You are a strict answer evaluator. "
    "Given a question, a model's predicted answer, and the gold standard answer, "
    "determine whether the predicted answer is semantically equivalent to the gold answer. "
    "Reply ONLY with a JSON object in the format:\n"
    '{"is_correct": true/false, "reason": "<one sentence explanation>"}'
)

def judge_with_deepseek(question: str, prediction: str, gold: str) -> dict:
    """调用 DeepSeek API 判断语义等价性。"""
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

# ─────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────
def build_prompt(sample: dict) -> str:
    """从样本构建用于生成的 prompt。"""
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
    """从 JSON 或 JSONL 文件加载样本。"""
    if not os.path.exists(path):
        print(f"错误: 输入文件不存在: {path}")
        return []
        
    with open(path, "r", encoding="utf-8") as fin:
        raw = fin.read().strip()

    if not raw:
        return []

    if raw.startswith("["):
        return json.loads(raw)

    samples = []
    for i, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[跳过] 第 {i} 行 JSON 解析失败: {e}")
    return samples

# ─────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────
print("正在加载 Tokenizer 和原始 Qwen3 模型...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = Qwen3ForCausalLM.from_pretrained(
    args.model_path,
    attn_implementation="flash_attention_2",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

def generate_answer(prompt: str) -> str:
    """使用加载的模型生成答案。"""
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0,
    }
    
    generated_ids = model.generate(input_ids, **gen_kwargs)

    return tokenizer.decode(
        generated_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()

# ─────────────────────────────────────────
# Main Evaluation Loop
# ─────────────────────────────────────────
print(f"正在从以下路径读取数据: {args.input_jsonl}")
samples = load_samples(args.input_jsonl)
print(f"已加载 {len(samples)} 个样本进行重新评估。")

total = correct = wrong = 0

with open(args.output_jsonl, "w", encoding="utf-8") as fout:
    for sample in tqdm(samples, desc="使用原始 Qwen3 重新评估"):
        question = sample.get("question", "")
        gold_answer = sample.get("answer", "") or sample.get("gold_answer", "")

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
        
        # 无论对错，都记录下来以供分析
        record = {
            "id": sample.get("id", total),
            "question": question,
            "gold_answer": gold_answer,
            "prediction": prediction,
            "is_correct_now": is_correct,
            "judge_reason": reason,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        tqdm.write(
            f"[{total}] correct={correct} wrong={wrong} | "
            f"{'✓' if is_correct else '✗'} | {reason}"
        )

# ─────────────────────────────────────────
# Summary
# ─────────────────────────────────────────
print("\n========== 重新评估总结 ==========")
print(f"总样本数 : {total}")
if total > 0:
    print(f"正确数   : {correct}  ({correct / total * 100:.1f}%)")
    print(f"错误数   : {wrong}  ({wrong / total * 100:.1f}%)")
else:
    print("正确数   : 0")
    print("错误数   : 0")
print(f"重新评估结果已保存至: {args.output_jsonl}")
