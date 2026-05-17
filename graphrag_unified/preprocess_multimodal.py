"""Preprocess manuals: preserve <PIC> markers and record image positions in chunks."""
import json
import os
import re

INPUT_DIR = "/home/wyz/kefu/data/手册"
OUTPUT_DIR = "/home/wyz/kefu/graphrag_unified/input"
CHUNK_META_FILE = "/home/wyz/kefu/graphrag_unified/chunk_metadata.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def process_manual(filepath):
    """Parse manual JSON and extract text + image list.

    Supports two formats:
    1. Single JSON array: ["text", ["image1", "image2"]]
    2. Multi-line format: each line is a separate JSON array (for merged manuals)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # Try single JSON first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) >= 2:
            text = parsed[0]
            images = parsed[1] if isinstance(parsed[1], list) else []
            # Replace literal \n with actual newlines (keep <PIC>)
            text = text.replace("\\n", "\n")
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()
            return text, images
    except json.JSONDecodeError:
        pass

    # Multi-line format: each line is a separate JSON array
    all_text = []
    all_images = []

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, list) and len(parsed) >= 2:
                text = parsed[0]
                images = parsed[1] if isinstance(parsed[1], list) else []
                all_text.append(text)
                all_images.extend(images)
        except json.JSONDecodeError:
            continue

    if all_text:
        text = "\n".join(all_text)
        text = text.replace("\\n", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text, all_images

    # Fallback: treat as plain text
    text = raw.replace("\\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text, []


def find_pic_positions(text, images):
    """Find all <PIC> positions and map to image IDs."""
    pic_list = []
    img_idx = 0

    for match in re.finditer(r"<PIC>", text):
        pos = match.start()
        image_id = images[img_idx] if img_idx < len(images) else None
        pic_list.append({
            "pos": pos,
            "image_id": image_id,
            "end": match.end()
        })
        img_idx += 1

    return pic_list


def split_text_with_pic(text, images, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into chunks, recording <PIC> positions."""
    chunks = []
    pic_positions = find_pic_positions(text, images)
    pic_idx = 0

    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Find all <PIC> markers in this chunk
        chunk_pics = []
        while pic_idx < len(pic_positions) and pic_positions[pic_idx]["pos"] < end:
            if pic_positions[pic_idx]["pos"] >= start:
                chunk_pics.append(pic_positions[pic_idx])
            pic_idx += 1

        chunk_text = text[start:end]

        # Get context around each <PIC> for later image selection
        pic_info = []
        for pic in chunk_pics:
            ctx_start = max(0, pic["pos"] - start - 150)
            ctx_end = min(len(chunk_text), pic["pos"] - start + 150)
            context = chunk_text[ctx_start:ctx_end].strip()
            pic_info.append({
                "pos_in_chunk": pic["pos"] - start,
                "image_id": pic["image_id"],
                "context": context
            })

        chunks.append({
            "text": chunk_text,
            "pics": pic_info
        })

        # Move to next chunk with overlap
        start += chunk_size - overlap
        # Reset pic_idx to handle overlap
        pic_idx = max(0, pic_idx - 1)

    return chunks


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metadata = {}

    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.endswith(".txt"):
            continue

        src = os.path.join(INPUT_DIR, fname)
        text, images = process_manual(src)

        # Find all <PIC> positions with image mapping
        pic_list = find_pic_positions(text, images)

        # Split into chunks
        chunks = split_text_with_pic(text, images)

        # Write processed text (with <PIC> markers)
        dst = os.path.join(OUTPUT_DIR, fname)
        with open(dst, "w", encoding="utf-8") as f:
            f.write(text)

        # Store metadata
        manual_name = fname.replace(".txt", "")
        all_metadata[manual_name] = {
            "total_images": len(images),
            "total_pics": len(pic_list),
            "chunks": chunks
        }

        # Count chunks with images
        chunks_with_pics = sum(1 for c in chunks if c["pics"])

        print(f"Processed: {fname}")
        print(f"  Text length: {len(text)} chars")
        print(f"  <PIC> markers: {len(pic_list)}")
        print(f"  Images: {len(images)}")
        print(f"  Chunks: {len(chunks)} (with pics: {chunks_with_pics})")

    # Save metadata
    with open(CHUNK_META_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"\nChunk metadata saved to: {CHUNK_META_FILE}")
    print(f"Total manuals: {len(all_metadata)}")


if __name__ == "__main__":
    main()
