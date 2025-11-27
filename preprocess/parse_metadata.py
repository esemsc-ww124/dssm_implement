# preprocess/parse_metadata.py
import gzip
import json

def parse(path):
    with gzip.open(path, 'r') as f:
        for line in f:
            yield json.loads(line)

def clean_description(desc):
        if isinstance(desc, list):
            return " ".join(desc)
        if isinstance(desc, str):
            return desc
        return ""

def clean_category(cat):
        if isinstance(cat, list):
            return cat
        return []

def build_item_meta(meta_path):
    item_meta = {}

    for meta in parse(meta_path):
        asin = meta.get("asin")
        if asin is None:
            continue

        title = meta.get("title", "")
        description = clean_description(meta.get("description", ""))
        categories = clean_category(meta.get("category", []))

        category = categories[0] if len(categories) > 0 else "unknown"

        item_meta[asin] = {
            "title": title,
            "description": description,
            "category": category
        }

    return item_meta
