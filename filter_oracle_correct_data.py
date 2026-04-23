#!/usr/bin/env python3
"""
过滤 HotpotQA 数据，只保留“自实现 Qwen 模型 + Oracle Context”下可答对的样本。

流程:
1) 读取 HotpotQA JSON
2) 从 supporting_facts 构造仅证据文章的 oracle context
3) 调用 [test_qwen_contribute.py](http://_vscodecontentref_/9) 里的自实现模型推理
4) 用 DeepSeek 裁判做语义等价判定
5) 正确样本写入 JSONL
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # 只使用第一张 GPU
import random
import sys
import time
from typing import Any, Dict, List, Set, Tuple

from openai import OpenAI

# 只使用你自己实现模型的公开入口
from test_qwen_contribute import load_qwen_runtime, generate_answer


def load_hotpotqa(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # 兼容常见封装格式
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        # 极端情况下字典的 value 里可能包含列表
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    raise ValueError(f"无法识别的数据格式: {path}")


def load_processed_ids(output_file: str) -> Set[str]:
    processed = set()
    if not os.path.exists(output_file):
        return processed

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            qid = obj.get("id") or obj.get("question_id")
            if qid is not None:
                processed.add(str(qid))
    return processed


def get_supporting_titles(item: Dict[str, Any]) -> Set[str]:
    sf = item.get("supporting_facts", [])
    titles: Set[str] = set()

    # 常见格式1: [["title", sent_idx], ...]
    if isinstance(sf, list):
        for row in sf:
            if isinstance(row, list) and row:
                titles.add(str(row[0]))
            elif isinstance(row, dict):
                t = row.get("title")
                if t is not None:
                    titles.add(str(t))

    # 常见格式2: {"title": [...], "sent_id": [...]}
    if isinstance(sf, dict):
        t = sf.get("title", [])
        if isinstance(t, list):
            titles.update(str(x) for x in t)

    return titles


def parse_context_pairs(item: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    ctx = item.get("context", [])
    pairs: List[Tuple[str, List[str]]] = []

    # 格式A: {"title":[...], "sentences":[[...], ...]}
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sents_all = ctx.get("sentences", [])
        for t, sents in zip(titles, sents_all):
            if not isinstance(sents, list):
                sents = [str(sents)]
            pairs.append((str(t), [str(x) for x in sents]))
        return pairs

    # 格式B: [[title, [sent1, sent2...]], ...]
    if isinstance(ctx, list):
        for row in ctx:
            if not isinstance(row, list) or len(row) < 2:
                continue
            title = str(row[0])
            sents = row[1]
            if not isinstance(sents, list):
                sents = [str(sents)]
            pairs.append((title, [str(x) for x in sents]))

    return pairs


def build_oracle_context(item: Dict[str, Any], rng: random.Random) -> str:
    support_titles = get_supporting_titles(item)
    pairs = parse_context_pairs(item)

    selected = [(title, sents) for title, sents in pairs if title in support_titles]
    if not selected:
        return ""

    rng.shuffle(selected)

    blocks: List[str] = []
    for title, sents in selected:
        lines = [f"Title: {title}"]
        for s in sents:
            s = s.strip()
            if s:
                lines.append(s)
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


class DeepSeekJudge:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout: int = 120,
        max_retries: int = 3,
        temperature: float = 0.0,
    ) -> None:
        # 兼容用户传入 /chat/completions 或根地址
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            normalized = normalized[: -len("/chat/completions")]

        self.client = OpenAI(api_key=api_key, base_url=normalized, timeout=timeout)
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature

    def evaluate(self, question: str, gold: str, pred: str) -> Dict[str, Any]:
        prompt = (
            "You are a strict QA evaluator.\n"
            f"Question: {question}\n"
            f"Gold Answer: {gold}\n"
            f"Predicted Answer: {pred}\n\n"
            "Judge whether predicted answer is semantically equivalent to gold answer.\n"
            "Output ONLY JSON with keys:\n"
            '  "is_correct": boolean,\n'
            '  "reason": string\n'
        )

        last_err = ""
        for _ in range(self.max_retries):
            try:
                print("正在调用 DeepSeek 评判...", flush=True)
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                print("DeepSeek 评判完成", flush=True)
                content = resp.choices[0].message.content or "{}"
                obj = json.loads(content)
                return {
                    "is_correct": bool(obj.get("is_correct", False)),
                    "reason": str(obj.get("reason", "")),
                }
            except Exception as e:
                last_err = str(e)
                print(f"[WARN] DeepSeek 评判失败: {last_err}，正在重试...", flush=True)
                time.sleep(1.0)

        return {"is_correct": False, "reason": f"judge_failed: {last_err}"}


def evaluate_single_run(
    runtime: Dict[str, Any],
    judge: DeepSeekJudge,
    question: str,
    gold: str,
    context_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    quiet_model_stdout: bool = True,
) -> Dict[str, Any]:
    # [test_qwen_contribute.py](http://_vscodecontentref_/10) 里 generate 默认会打印大量日志，这里静默掉
    try:
        if quiet_model_stdout:
            with contextlib.redirect_stdout(io.StringIO()):
                pred_answer = generate_answer(
                    runtime=runtime,
                    question=question,
                    context=context_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
        else:
            pred_answer = generate_answer(
                runtime=runtime,
                question=question,
                context=context_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        print(pred_answer)
    except Exception as e:
        raise RuntimeError(f"model_inference_failed: {e}") from e

    judge_result = judge.evaluate(question=question, gold=gold, pred=pred_answer)
    return {
        "pred_answer": pred_answer,
        "is_correct": judge_result.get("is_correct", False),
        "judge": judge_result,
    }


def run_filter(args: argparse.Namespace) -> None:
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    deepseek_api_key = args.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise ValueError("缺少 DeepSeek API Key。请传 --deepseek-api-key 或设置 DEEPSEEK_API_KEY。")

    data = load_hotpotqa(args.data)
    if args.limit and args.limit > 0:
        data = data[: args.limit]
    print(f"[INFO] 共加载 {len(data)} 条样本进行过滤", flush=True)

    processed_ids = load_processed_ids(args.out) if args.resume else set()
    if processed_ids:
        print(f"[INFO] 断点续跑：已跳过 {len(processed_ids)} 条已处理样本", flush=True)

    # 使用自实现模型加载
    runtime = load_qwen_runtime(
        model_path=args.model,
        dtype=args.dtype,
        device=args.device,
    )

    judge = DeepSeekJudge(
        api_key=deepseek_api_key,
        model=args.deepseek_judge_model,
        base_url=args.deepseek_base_url,
        timeout=args.deepseek_timeout,
        max_retries=args.deepseek_max_retries,
        temperature=args.deepseek_temperature,
    )

    total_processed = 0
    kept_samples = 0
    t0 = time.time()

    with open(args.out, "a", encoding="utf-8") as fout:
        for idx, item in enumerate(data):
            qid = str(item.get("id", f"idx_{idx}"))
            if qid in processed_ids:
                continue

            question = str(item.get("question", "")).strip()
            gold = str(item.get("answer", "")).strip()
            if not question or not gold:
                print(f"[WARN] id={qid} question/answer 缺失，跳过", flush=True)
                continue

            oracle_rng = random.Random(args.seed + idx)
            oracle_context_text = build_oracle_context(item, oracle_rng)
            if not oracle_context_text:
                print(f"[WARN] id={qid} 未构造出证据上下文，跳过", flush=True)
                continue

            try:
                print("开始评测 id={}...".format(qid), flush=True)
                result = evaluate_single_run(
                    runtime=runtime,
                    judge=judge,
                    question=question,
                    gold=gold,
                    context_text=oracle_context_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    quiet_model_stdout=args.quiet_model_stdout,
                )
            except Exception as e:
                print(f"[ERROR] id={qid} 评测失败: {e}", flush=True)
                continue

            total_processed += 1

            if result.get("is_correct"):
                kept_samples += 1
                record = item.copy()
                record["_filter_info"] = {
                    "is_correct_on_oracle": True,
                    "prediction": result.get("pred_answer"),
                    "judge_reason": result.get("judge", {}).get("reason"),
                    "model_impl": "test_qwen_contribute.load_qwen_runtime + generate_answer",
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

            if (idx + 1) % args.log_every == 0:
                elapsed = time.time() - t0
                ratio = kept_samples / total_processed if total_processed else 0.0
                print(
                    f"[{idx + 1}/{len(data)}] 已处理={total_processed} 已保留={kept_samples} "
                    f"保留率={ratio:.2%} 耗时={elapsed:.1f}s",
                    flush=True,
                )

    elapsed = time.time() - t0
    print("-" * 80, flush=True)
    print(f"[DONE] 总共检查了 {total_processed} 个样本", flush=True)
    print(f"[DONE] 保留了 {kept_samples} 个样本", flush=True)
    if total_processed > 0:
        print(f"[DONE] 保留率: {kept_samples / total_processed:.2%}", flush=True)
    print(f"[DONE] 总耗时: {elapsed:.1f}s", flush=True)
    print(f"[DONE] 过滤后的数据已保存到: {args.out}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="过滤 HotpotQA 数据，只保留在 Oracle Context 下能答对的样本（自实现Qwen推理）。"
    )

    p.add_argument("--data", required=True, help="输入 HotpotQA JSON 文件路径")
    p.add_argument("--out", required=True, help="输出 JSONL 文件路径")
    p.add_argument("--model", required=True, help="本地模型目录（包含 config.json 和权重）")

    p.add_argument("--limit", type=int, default=0, help="限制处理样本数，0=全部")
    p.add_argument("--log-every", type=int, default=20, help="每 N 条打印日志")
    p.add_argument("--resume", action="store_true", help="断点续跑，跳过输出中已有ID")
    p.add_argument("--seed", type=int, default=42, help="oracle 证据文章乱序种子")

    p.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0 或 cuda:0,cuda:1")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--cuda-visible-devices", default=None, help="覆盖 CUDA_VISIBLE_DEVICES")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--quiet-model-stdout", action="store_true", help="静默自实现模型的过程日志")

    p.add_argument("--deepseek-api-key", default="sk-351bc73eb0d24bff98501cd30f902836", help="DeepSeek API Key 或环境变量 DEEPSEEK_API_KEY")
    p.add_argument("--deepseek-judge-model", default="deepseek-chat")
    p.add_argument("--deepseek-base-url", default="https://api.deepseek.com/v1")
    p.add_argument("--deepseek-timeout", type=int, default=120)
    p.add_argument("--deepseek-max-retries", type=int, default=3)
    p.add_argument("--deepseek-temperature", type=float, default=0.0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_filter(args)
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，已保存当前进度。", flush=True)
        sys.exit(130)