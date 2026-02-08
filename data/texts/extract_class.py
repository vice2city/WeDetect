import json
import argparse
from pathlib import Path


def load_coco_categories(coco_json_path: Path):
    with coco_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "categories" not in data or not isinstance(data["categories"], list):
        raise ValueError("输入不是有效的 COCO 标注文件：缺少 categories 列表")
    # 按 id 排序，输出稳定
    cats = sorted(data["categories"], key=lambda x: x.get("id", 0))
    names = [c["name"] for c in cats if "name" in c]
    if not names:
        raise ValueError("categories 中未找到 name 字段")
    return names


def save_uodd_class_texts(names, output_path: Path):
    arr = [[n] for n in names]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="从 COCO 标注生成 uodd_class_texts.json")
    parser.add_argument("coco_json", type=str, help="COCO 标注 JSON 文件路径")
    parser.add_argument("output_json", type=str, help="输出 uodd_class_texts.json 文件路径")
    args = parser.parse_args()

    coco_path = Path(args.coco_json)
    out_path = Path(args.output_json)

    names = load_coco_categories(coco_path)
    save_uodd_class_texts(names, out_path)
    print(f"已写入 {out_path}，类别数：{len(names)}")


if __name__ == "__main__":
    main()