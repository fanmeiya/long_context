import json, yaml, torch, os, argparse
from tqdm import tqdm
from openai import OpenAI

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from transformers import AutoTokenizer
from dysco.custom_modeling_qwen3 import RescaleQwen3ForCausalLM
from dysco.custom_mixin import RescaleConfig

# ─────────────────────────────────────────
# Args
# ─────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input_jsonl",  required=True, help="Path to the input .jsonl file")
parser.add_argument("--output_jsonl", default="wrong_samples.jsonl", help="Path to save wrong samples")
parser.add_argument("--model_path",   default="/home/Datasets/Hf_model/Qwen3-8B/")
parser.add_argument("--cfg_path",     default="dysco_cfgs/qwen3_8b.yaml")
parser.add_argument("--deepseek_api_key", default=os.environ.get("DEEPSEEK_API_KEY", ""), help="DeepSeek API key")
parser.add_argument("--deepseek_base_url", default="https://api.deepseek.com", help="DeepSeek base URL")
parser.add_argument("--max_new_tokens", type=int, default=512)
args = parser.parse_args()

# ─────────────────────────────────────────
# DeepSeek judge client
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
    """Call DeepSeek API to judge semantic equivalence. Returns dict with is_correct & reason."""
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
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
        # strip possible markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        return {"is_correct": False, "reason": f"Judge API error: {e}"}

# ─────────────────────────────────────────
# Load model & tokenizer
# ─────────────────────────────────────────
print("Loading tokenizer and model …")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = RescaleQwen3ForCausalLM.from_pretrained(
    args.model_path,
    attn_implementation="flash_attention_2",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# ─────────────────────────────────────────
# Build RescaleConfig
# ─────────────────────────────────────────
with open(args.cfg_path) as f:
    cfg = yaml.safe_load(f)

selected_heads = eval(cfg["selected_heads"])
intervention_warmup_steps = cfg.get("intervention_warmup_steps", 0)
if intervention_warmup_steps in ("auto", None):
    intervention_warmup_steps = 0

rescale_config = RescaleConfig(
    selected_heads=selected_heads,
    top_k=cfg["top_k"], top_p=cfg["top_p"],
    strength=cfg["strength"], decay_factor=cfg["decay_factor"],
    context_warmup_steps=cfg.get("context_warmup_steps", 0),
    intervention_warmup_steps=intervention_warmup_steps,
)

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def build_prompt(sample: dict) -> str:
    """
    Construct the user-facing prompt from a JSONL sample.
    Supports two common formats:
      1. {"question": "...", "context": {"title": [...], "sentences": [...]}}
      2. {"question": "...", "context": "plain text"}
    Falls back gracefully if fields are missing.
    """
    question = sample.get("question", "")
    context_raw = sample.get("context", None)

    if context_raw is None:
        context_str = ""
    elif isinstance(context_raw, str):
        context_str = context_raw
    elif isinstance(context_raw, dict):
        # HotpotQA-style structured context
        titles    = context_raw.get("title", [])
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
    else:
        return f"Answer the following question.\n\nQuestion: {question}"


def generate_answer(prompt: str) -> str:
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    generated_ids, _ = model.rescale_generate(
        input_ids,
        rescale_config=rescale_config,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(
        generated_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()

# ─────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────
total = correct = wrong = 0
wrong_samples = []

# ─────────────────────────────────────────
# Load input (兼容 JSON array 和 JSONL 两种格式)
# ─────────────────────────────────────────
print(f"Reading data from: {args.input_jsonl}")
with open(args.input_jsonl, "r", encoding="utf-8") as fin:
    raw = fin.read().strip()

if raw.startswith("["):
    # JSON array 格式: [{...}, {...}, ...]
    samples = json.loads(raw)
else:
    # JSONL 格式: 每行一个 JSON 对象
    samples = []
    for i, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[跳过] 第 {i} 行 JSON 解析失败: {e}")

print(f"共加载 {len(samples)} 条样本")

# ─────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────
total = correct = wrong = 0

with open(args.output_jsonl, "w", encoding="utf-8") as fout:
    for sample in tqdm(samples, desc="Evaluating"):
        question    = sample.get("question", "")
        gold_answer = sample.get("answer", "")

        # Generate model prediction
        prompt     = build_prompt(sample)
        prediction = generate_answer(prompt)

        # Judge with DeepSeek
        judge      = judge_with_deepseek(question, prediction, gold_answer)
        is_correct = judge.get("is_correct", False)
        reason     = judge.get("reason", "")

        total += 1
        if is_correct:
            correct += 1
        else:
            wrong += 1
            record = {
                "id":           sample.get("id", total),
                "question":     question,
                "gold_answer":  gold_answer,
                "prediction":   prediction,
                "judge_reason": reason,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Live progress
        tqdm.write(
            f"[{total}] correct={correct} wrong={wrong} | "
            f"{'✓' if is_correct else '✗'} | {reason}"
        )

# ─────────────────────────────────────────
# Summary
# ─────────────────────────────────────────
print("\n========== Evaluation Summary ==========")
print(f"Total samples : {total}")
print(f"Correct       : {correct}  ({correct/total*100:.1f}%)")
print(f"Wrong         : {wrong}  ({wrong/total*100:.1f}%)")
print(f"Wrong samples saved to: {args.output_jsonl}")