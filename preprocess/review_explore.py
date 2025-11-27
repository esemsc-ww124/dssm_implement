# ===== 字段缺失程度（前 5 万条）=====
# reviewerID: 0 / 50000 (0.00%)
# asin: 0 / 50000 (0.00%)
# overall: 0 / 50000 (0.00%)
# vote: 45132 / 50000 (90.26%)
# reviewerID, asin, overall
import gzip
import json
from collections import Counter

path = "../data/Books.json.gz"
MAX_LINES = 50000  # 你可以改成更大

FIELDS = ["reviewerID", "asin", "overall", "vote"]
missing_counter = Counter()
total = 0

def parse(path):
    with gzip.open(path, "r") as f:
        for line in f:
            yield json.loads(line)

print("开始统计前 5 万条 review 字段缺失情况...\n")

for obj in parse(path):
    total += 1

    for field in FIELDS:
        value = obj.get(field, None)

        # 判定缺失：字段不存在 或者 值为空
        if value is None or value == "" or value == []:
            missing_counter[field] += 1

    if total >= MAX_LINES:
        break

# 输出结果
print("===== 字段缺失程度（前 5 万条）=====")
for field in FIELDS:
    miss = missing_counter[field]
    print(f"{field}: {miss} / {total} ({miss / total * 100:.2f}%)")
