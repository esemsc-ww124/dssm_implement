# preprocess/calc_rating_stats.py
import gzip
import json
import pickle
from collections import defaultdict

def parse(path):
    with gzip.open(path, 'r') as f:
        for line in f:
            yield json.loads(line)

def calc_rating_stats(reviews_path):
    item_sum = defaultdict(float)
    item_cnt = defaultdict(int)

    for r in parse(reviews_path):
        asin = r.get("asin")
        score = r.get("overall")

        if not asin or score is None:
            continue

        item_sum[asin] += float(score)
        item_cnt[asin] += 1

    item_stats = {
        asin: {
            "overall_mean": item_sum[asin] / item_cnt[asin],
            "review_count": item_cnt[asin]
        }
        for asin in item_sum.keys()
    }

    return item_stats
