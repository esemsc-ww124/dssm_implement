# preprocess/parse_reviews.py
import gzip
import json

def parse(path):
    with gzip.open(path, 'r') as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                # 新版可能不是严格 json，用 eval 兜底
                yield eval(line)

def build_user_sequences(reviews_path):
    user_interactions = {}

    for review in parse(reviews_path):
        # 必备字段检查
        if ("reviewerID" not in review) or ("asin" not in review):
            continue

        user = review["reviewerID"]
        item = review["asin"]

        # 有些新版 review 没有 unixReviewTime，用 reviewTime 转换
        ts = review.get("unixReviewTime")

        if ts is None:
            # 兼容 "01 1, 2018" 格式
            import time
            try:
                ts = int(time.mktime(time.strptime(review["reviewTime"], "%m %d, %Y")))
            except:
                continue

        if user not in user_interactions:
            user_interactions[user] = []

        user_interactions[user].append((ts, item))

    # 排序
    sorted_sequences = {
        user: [item for _, item in sorted(pairs)]
        for user, pairs in user_interactions.items()
    }

    return sorted_sequences
