import argparse
import json
import os
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Split a dataset into train/validation/test.')
    parser.add_argument('--input', '-I', type=str, required=True, help='Input JSONL file.')
    parser.add_argument('--portion', '-P', type=str, default='90:5:5', help='Split ratio.')
    parser.add_argument('--output_dir', '-O', type=str, default="./data", help='Output directory.')
    parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the dataset before splitting.')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.input, 'r') as f:
        data = [json.loads(line) for line in f]

    # 随机打乱数据（如果指定了 --shuffle 参数）
    if args.shuffle:
        random.shuffle(data)

    # 解析划分比例
    portions = list(map(int, args.portion.split(':')))
    total = sum(portions)
    train_ratio, val_ratio, test_ratio = [p / total for p in portions]

    # 计算每个部分的数据量
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)

    # 划分数据集
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # 创建输出目录（如果不存在）
    os.makedirs(args.output_dir, exist_ok=True)

    # 将划分后的数据写入输出文件
    with open(os.path.join(args.output_dir, 'train.jsonl'), 'w') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(os.path.join(args.output_dir, 'validation.jsonl'), 'w') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(os.path.join(args.output_dir, 'test.jsonl'), 'w') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Dataset split completed. Output files: {args.output_dir}/{{train,val,test}}.jsonl")

if __name__ == '__main__':
    main()