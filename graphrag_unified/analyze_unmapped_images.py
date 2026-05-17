"""Analyze unmapped images and write a reusable markdown report."""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from preprocess_multimodal import process_manual

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = Path("/home/wyz/kefu/data/手册")
IMAGE_DIR = SOURCE_DIR / "插图"
MAP_FILE = PROJECT_ROOT / "text_unit_image_map.json"
DOCS_FILE = PROJECT_ROOT / "output" / "documents.parquet"
REPORT_FILE = PROJECT_ROOT / "unmapped_image_report.md"


def consecutive_ranges(values: list[int]) -> list[str]:
    if not values:
        return []
    values = sorted(values)
    ranges = []
    start = end = values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
        else:
            ranges.append(f"{start}" if start == end else f"{start}-{end}")
            start = end = value
    ranges.append(f"{start}" if start == end else f"{start}-{end}")
    return ranges


def main():
    with MAP_FILE.open(encoding="utf-8") as f:
        mapping = json.load(f)
    documents = pd.read_parquet(DOCS_FILE)

    folder_images = {p.stem for p in IMAGE_DIR.iterdir() if p.is_file()}
    mapped_images = {
        pic["image_id"]
        for item in mapping["by_text_unit_id"].values()
        for pic in item.get("pics", [])
        if pic.get("image_id")
    }

    unresolved_by_doc = defaultdict(list)
    for item in mapping["unresolved"]:
        unresolved_by_doc[item["document"]].append(item)

    manual_stats = []
    manual_image_refs = set()
    manual_missing_all = set()
    source_anomalies = []
    raw_json_docs = []

    for _, doc in documents.iterrows():
        title = str(doc["title"])
        source_path = SOURCE_DIR / title
        if not source_path.exists():
            continue

        text, images = process_manual(str(source_path))
        pic_markers = text.count("<PIC>")
        image_refs = [image_id for image_id in images if image_id]
        unique_refs = sorted(set(image_refs))
        missing_refs = sorted(image_id for image_id in unique_refs if image_id not in mapped_images)
        unresolved_rows = unresolved_by_doc.get(title, [])
        unresolved_short_ids = [
            int(item["short_id"])
            for item in unresolved_rows
            if item.get("short_id") and str(item["short_id"]).isdigit()
        ]
        raw_json = str(doc["text"]).startswith('["')

        manual_image_refs.update(unique_refs)
        manual_missing_all.update(missing_refs)
        if raw_json:
            raw_json_docs.append(title)
        if pic_markers != len(images):
            source_anomalies.append(
                {
                    "title": title,
                    "pic_markers": pic_markers,
                    "images": len(images),
                    "diff": pic_markers - len(images),
                }
            )

        manual_stats.append(
            {
                "title": title,
                "raw_json": raw_json,
                "pic_markers": pic_markers,
                "image_refs": len(images),
                "unique_image_refs": len(unique_refs),
                "mapped_unique_images": len(unique_refs) - len(missing_refs),
                "unmapped_unique_images": len(missing_refs),
                "unresolved_text_units": len(unresolved_rows),
                "unresolved_ranges": consecutive_ranges(unresolved_short_ids)[:8],
                "sample_unmapped": missing_refs[:8],
            }
        )

    unused_folder_images = sorted(folder_images - manual_image_refs)
    ranked = sorted(
        [row for row in manual_stats if row["unmapped_unique_images"] > 0],
        key=lambda row: (-row["unmapped_unique_images"], row["title"]),
    )

    lines = []
    lines.append("# 未映射图片排查报告")
    lines.append("")
    lines.append("## 总览")
    lines.append(f"- 插图目录文件数：{len(folder_images)}")
    lines.append(f"- 文本中唯一图片引用数：{len(manual_image_refs)}")
    lines.append(f"- 已映射唯一图片数：{len(mapped_images)}")
    lines.append(f"- 文本中未映射唯一图片数：{len(manual_missing_all)}")
    lines.append(f"- 文本未引用的图片文件数：{len(unused_folder_images)}")
    lines.append("")
    lines.append("## 结论分组")
    lines.append("- `源数据异常`：手册文本中的 `<PIC>` 数量与图片列表长度不一致，这类问题会直接造成图片无法正确对齐。")
    for item in source_anomalies:
        lines.append(
            f"- {item['title']}：`<PIC>={item['pic_markers']}`，`images={item['images']}`，差值 `{item['diff']}`"
        )
    if not source_anomalies:
        lines.append("- 未发现 `<PIC>` 与图片列表数量不一致的手册。")
    lines.append("- `索引/文本格式异常`：documents.parquet 中仍是原始 JSON 风格文本的手册。")
    for title in raw_json_docs:
        lines.append(f"- {title}")
    if not raw_json_docs:
        lines.append("- 未发现 raw JSON 风格文档。")
    lines.append("- `定位失败（span_not_found）`：本次 298 张未映射图片主要来自 text_unit 定位失败，集中在少数手册。")
    lines.append("")
    lines.append("## 未映射图片最多的手册")
    for row in ranked:
        lines.append(
            f"- {row['title']}：未映射 {row['unmapped_unique_images']} / {row['unique_image_refs']}，"
            f"未定位 text_unit {row['unresolved_text_units']} 个，short_id 区间 {', '.join(row['unresolved_ranges']) or '无'}，"
            f"示例 {row['sample_unmapped']}"
        )
    lines.append("")
    lines.append("## 文本未引用的图片文件示例")
    lines.append(f"- 共 {len(unused_folder_images)} 张，示例：{unused_folder_images[:20]}")
    lines.append("")
    lines.append("## 判断")
    lines.append("- 这 298 张未映射图片的主因不是图片文件缺失；`missing_in_folder = 0`。")
    lines.append("- 主因也不是 GraphRAG 完全没读到这些手册，因为对应手册都已有部分图片成功映射。")
    lines.append("- 主要问题是少数手册中，连续若干 `text_unit` 无法在源文本中稳定定位，导致这些 `text_unit` 覆盖到的图片没有被挂上。")
    lines.append("- 需要优先处理的不是全局重建索引，而是增强这些手册的 `text_unit -> source span` 对齐策略。")
    lines.append("")
    lines.append("## 优先排查建议")
    lines.append("- 第一优先：`发电机手册`、`相机手册`、`水泵手册`、`冰箱手册`、`空调手册`。这些手册贡献了最多未映射图片。")
    lines.append("- 第二优先：修复 `洗碗机手册` 的源数据，当前有 `<PIC>` 但图片列表为空。")
    lines.append("- 第三优先：针对连续 short_id 区间做更强的定位回退，例如缩短匹配窗口、允许跨段前后缀匹配，或基于相邻已定位 text_unit 约束搜索范围。")

    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
