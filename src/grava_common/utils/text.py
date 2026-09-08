"""Text processing utilities."""
import re
from typing import Optional


def extract_tag_content(text: str, tag: str) -> Optional[str]:
    """Extract content between XML-like tags."""
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def has_tags(text: str, tags: list[str]) -> bool:
    """Check if text contains all specified tags."""
    lowered = text.lower()
    return all(f"<{tag}>" in lowered for tag in tags)
