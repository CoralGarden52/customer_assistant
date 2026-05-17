"""Utilities for multimodal image lookup and selection."""
import json
import re
from pathlib import Path

CHUNK_META_FILE = Path(__file__).parent / "chunk_metadata.json"


def normalize_chunk_text(text: str) -> str:
    """Normalize chunk text for robust metadata matching."""
    if not text:
        return ""
    text = text.replace("\\u201c", '"').replace("\\u201d", '"')
    text = text.replace("\\u2018", "'").replace("\\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class ImageMapper:
    """Manages chunk metadata and provides image selection logic."""

    def __init__(self):
        self.metadata = {}
        if CHUNK_META_FILE.exists():
            with open(CHUNK_META_FILE, encoding="utf-8") as f:
                self.metadata = json.load(f)

    def get_chunk_images(self, chunk_text, manual_name=None):
        """Get images associated with a chunk based on text matching.

        Args:
            chunk_text: The chunk text to find images for
            manual_name: Optional manual name to limit search

        Returns:
            List of dicts with image_id and context
        """
        if not self.metadata:
            return []

        normalized_query = normalize_chunk_text(chunk_text)
        if not normalized_query:
            return []

        # Search through metadata to find matching chunk
        for mname, mdata in self.metadata.items():
            if manual_name and mname != manual_name:
                continue

            for chunk in mdata.get("chunks", []):
                metadata_text = chunk.get("text", "")
                if not metadata_text:
                    continue

                if metadata_text[:100] == chunk_text[:100]:
                    return chunk.get("pics", [])

                normalized_meta = normalize_chunk_text(metadata_text)
                query_prefix = normalized_query[:120]
                meta_prefix = normalized_meta[:120]
                if query_prefix and (query_prefix in normalized_meta or meta_prefix in normalized_query):
                    return chunk.get("pics", [])

        return []

    def format_image_info_for_llm(self, images):
        """Format image information for LLM prompt.

        Args:
            images: List of image dicts with image_id and context

        Returns:
            Formatted string for LLM
        """
        if not images:
            return ""

        lines = []
        for i, img in enumerate(images, 1):
            if img.get("image_id"):
                ctx = img.get("context", "").replace("\n", " ")[:100]
                lines.append(f"{i}. {img['image_id']}: {ctx}")

        return "\n".join(lines)

    def build_image_selection_prompt(self, question, chunk_text, images):
        """Build prompt for LLM to select relevant images.

        Args:
            question: User's question
            chunk_text: Retrieved chunk text
            images: List of image dicts

        Returns:
            Prompt string
        """
        image_info = self.format_image_info_for_llm(images)

        return f"""用户问题: {question}

检索到的内容:
{chunk_text[:500]}

该内容关联的图片及上下文:
{image_info}

请判断哪些图片与用户问题直接相关。
只返回与问题最相关的图片ID列表（JSON数组格式），不要返回所有图片。
如果没有相关图片，返回空数组 []。

示例输出: ["drill0_08", "drill0_09"] 或 []"""

    def select_relevant_images(self, question, chunk_text, images, llm_func):
        """Use LLM to select relevant images.

        Args:
            question: User's question
            chunk_text: Retrieved chunk text
            images: List of image dicts
            llm_func: Async function to call LLM

        Returns:
            List of relevant image IDs
        """
        if not images:
            return []

        # Filter out images without IDs
        valid_images = [img for img in images if img.get("image_id")]
        if not valid_images:
            return []

        # If only one image, return it directly
        if len(valid_images) == 1:
            return [valid_images[0]["image_id"]]

        # Use LLM to select
        prompt = self.build_image_selection_prompt(question, chunk_text, valid_images)

        try:
            response = llm_func(prompt)
            # Parse JSON array from response
            import re
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                selected = json.loads(match.group())
                # Validate image IDs
                valid_ids = {img["image_id"] for img in valid_images}
                return [img_id for img_id in selected if img_id in valid_ids]
        except Exception as e:
            print(f"    Image selection error: {e}", flush=True)

        # Fallback: return all valid images
        return [img["image_id"] for img in valid_images]

    def format_answer_with_images(self, answer, images):
        """Format answer with image references.

        Args:
            answer: The generated answer text
            images: List of image IDs

        Returns:
            Formatted answer string
        """
        if not images:
            return answer

        # Keep <PIC> markers in answer if present
        # Add image list at the end
        image_str = ", ".join(f'"{img}"' for img in images)
        return f'{answer}, [{image_str}]'


# Global instance
_image_mapper = None


def get_image_mapper():
    """Get or create the global ImageMapper instance."""
    global _image_mapper
    if _image_mapper is None:
        _image_mapper = ImageMapper()
    return _image_mapper
