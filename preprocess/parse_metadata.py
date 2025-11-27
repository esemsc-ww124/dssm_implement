# preprocess/parse_metadata.py
import gzip
import json

def parse(path):
    with gzip.open(path, 'r') as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                yield eval(line)

def build_item_meta(meta_path):
    item_meta = {}

    for meta in parse(meta_path):
        asin = meta.get("asin")
        if asin is None:
            continue

        title = meta.get("title", "")
        brand = meta.get("brand", "")
        categories = meta.get("categories", [[]])

        if categories and isinstance(categories[0], list):
            category = categories[0]
        else:
            category = categories

        item_meta[asin] = {
            "title": title,
            "brand": brand,
            "category": category
        }

    return item_meta
