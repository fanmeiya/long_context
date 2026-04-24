import json
import argparse
import os
from typing import Set, Dict, Any

def load_wrong_ids(filepath: str) -> Set[str]:
    """从 JSONL 文件中加载所有错误样本的 ID。"""
    ids = set()
    if not os.path.exists(filepath):
        print(f"警告: 文件不存在, 将视为空: {filepath}")
        return ids
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'id' in data:
                    ids.add(data['id'])
            except json.JSONDecodeError:
                print(f"警告: 解析行失败: {line.strip()}")
    print(f"从 {filepath} 加载了 {len(ids)} 个错误样本 ID。")
    return ids

def load_all_samples(filepath: str) -> Dict[str, Dict[str, Any]]:
    """从原始数据文件加载所有样本，并以 ID 为键创建字典。"""
    samples_dict = {}
    if not os.path.exists(filepath):
        print(f"错误: 原始数据文件不存在: {filepath}")
        return samples_dict
        
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            # 尝试作为 JSON array 读取
            data = json.load(f)
            for sample in data:
                if 'id' in sample:
                    samples_dict[sample['id']] = sample
        except json.JSONDecodeError:
            # 如果失败，尝试作为 JSONL 读取
            f.seek(0)
            for line in f:
                try:
                    sample = json.loads(line)
                    if 'id' in sample:
                        samples_dict[sample['id']] = sample
                except json.JSONDecodeError:
                    continue # 跳过无效行
    print(f"从 {filepath} 加载了 {len(samples_dict)} 个总样本。")
    return samples_dict

def save_samples_to_jsonl(samples: list, filepath: str):
    """将样本列表保存到 JSONL 文件。"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"已将 {len(samples)} 个样本保存到 {filepath}")

def main():
    parser = argparse.ArgumentParser(description="比较两个评估结果文件，并筛选出不同的错误样本。")
    parser.add_argument('--file1', type=str, default='wrong_samples.jsonl', help='第一个错误样本文件路径。')
    parser.add_argument('--file2', type=str, default='wrong_samples_vanilla.jsonl', help='第二个错误样本文件路径。')
    parser.add_argument('--source_data', type=str, default='test.json', help='包含所有样本的原始数据文件路径。')
    parser.add_argument('--output_dir', type=str, default='comparison_results', help='存放结果文件的目录。')
    
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载两份错误报告中的ID
    ids1 = load_wrong_ids(args.file1)
    ids2 = load_wrong_ids(args.file2)

    # 2. 计算ID集合
    common_wrong_ids = ids1.intersection(ids2)
    only_wrong_in_1_ids = ids1.difference(ids2)
    only_wrong_in_2_ids = ids2.difference(ids1)

    print(f"\n分析结果:")
    print(f"在 {os.path.basename(args.file1)} 和 {os.path.basename(args.file2)} 中都错误的样本数: {len(common_wrong_ids)}")
    print(f"仅在 {os.path.basename(args.file1)} 中错误的样本数: {len(only_wrong_in_1_ids)}")
    print(f"仅在 {os.path.basename(args.file2)} 中错误的样本数: {len(only_wrong_in_2_ids)}")

    # 3. 加载原始数据
    all_samples = load_all_samples(args.source_data)
    if not all_samples:
        print("无法加载原始样本，脚本终止。")
        return

    # 4. 根据ID筛选完整的样本数据
    common_wrong_samples = [all_samples[id] for id in common_wrong_ids if id in all_samples]
    only_wrong_in_1_samples = [all_samples[id] for id in only_wrong_in_1_ids if id in all_samples]
    only_wrong_in_2_samples = [all_samples[id] for id in only_wrong_in_2_ids if id in all_samples]

    # 5. 保存结果
    save_samples_to_jsonl(common_wrong_samples, os.path.join(args.output_dir, 'common_wrong.jsonl'))
    save_samples_to_jsonl(only_wrong_in_1_samples, os.path.join(args.output_dir, f'only_wrong_in_{os.path.splitext(os.path.basename(args.file1))[0]}.jsonl'))
    save_samples_to_jsonl(only_wrong_in_2_samples, os.path.join(args.output_dir, f'only_wrong_in_{os.path.splitext(os.path.basename(args.file2))[0]}.jsonl'))

    print("\n脚本执行完毕。")

if __name__ == '__main__':
    main()