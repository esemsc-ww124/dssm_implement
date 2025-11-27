# ===== 字段空值统计（前 5 万条）=====
# asin: 0 / 2934949 (0.00%)
# category: 389429 / 2934949 (13.27%)
# feature: 2933405 / 2934949 (99.95%)
# description: 550804 / 2934949 (18.77%)
# also_buy: 1590443 / 2934949 (54.19%)
# also_view: 1729794 / 2934949 (58.94%)
# title: 827 / 2934949 (0.03%)
# asin category description title overall(平均分) reviewcount(评分数量)

import gzip
import json
from collections import Counter

path = "../data/meta_Books.json.gz"
MAX_LINES = 500000

# 要检查的字段
TARGET_FIELDS = ["asin", "category", "feature", "description", "also_buy", "also_view", "title"]

# 统计空值
empty_count = Counter()
total = 0

# 保存前几个样本
samples = []

def clean_description(x):
    if isinstance(x, list):
        return " ".join(x)
    if isinstance(x, str):
        return x
    return ""

def clean_to_list(x):
    if isinstance(x, list):
        return x
    return []

with gzip.open(path, 'r') as f:
    for line in f:
        total += 1
        obj = json.loads(line)

        # 取字段
        asin = obj.get("asin", "")
        title = obj.get("title", "")

        category = clean_to_list(obj.get("category", []))
        feature = clean_to_list(obj.get("feature", []))

        description = clean_description(obj.get("description", ""))
        also_buy = clean_to_list(obj.get("also_buy", []))
        also_view = clean_to_list(obj.get("also_view", []))

        # 记录空值
        if asin == "":
            empty_count["asin"] += 1
        if title == "":
            empty_count["title"] += 1
        if category == []:
            empty_count["category"] += 1
        if feature == []:
            empty_count["feature"] += 1
        if description == "":
            empty_count["description"] += 1
        if also_buy == []:
            empty_count["also_buy"] += 1
        if also_view == []:
            empty_count["also_view"] += 1

        # 保存样本
        if len(samples) < 3:
            samples.append({
                "asin": asin,
                "title": title,
                "category": category,
                "feature": feature,
                "description": description,
                "also_buy": also_buy,
                "also_view": also_view
            })

# 输出结果
print("===== 字段空值统计（前 5 万条）=====")
for field in TARGET_FIELDS:
    missing = empty_count[field]
    print(f"{field}: {missing} / {total} ({missing/total*100:.2f}%)")

print("\n===== 示例样本（前 3 条）=====")
for s in samples:
    print(json.dumps(s, indent=2, ensure_ascii=False))
