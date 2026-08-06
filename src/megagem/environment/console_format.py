"""Shared rich-Text helpers for game-trace pretty-printing."""

from rich.text import Text

_GEM_COLORS = {
    "Red": "bright_red",
    "Blue": "bright_blue",
    "Green": "bright_green",
    "Purple": "bright_magenta",
    "Yellow": "bright_yellow",
}


def color_gem_name(text: str) -> Text:
    """Highlight gem names in ``text`` with rich color spans."""
    result = Text()
    remaining = text
    while remaining:
        earliest_pos = len(remaining)
        earliest_color = None
        earliest_name = None
        for gem_name, color in _GEM_COLORS.items():
            pos = remaining.find(gem_name)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
                earliest_color = color
                earliest_name = gem_name
        if earliest_color:
            if earliest_pos > 0:
                result.append(remaining[:earliest_pos])
            result.append(earliest_name, style=f"bold {earliest_color}")
            remaining = remaining[earliest_pos + len(earliest_name):]
        else:
            result.append(remaining)
            break
    return result


def format_gem_string_with_colors(gem_str: str) -> Text:
    """Format strings like ``'Red×2, Blue×1'`` with per-color highlighting."""
    if gem_str == "Empty":
        return Text("Empty", style="dim")
    result = Text()
    parts = gem_str.split(", ")
    for i, part in enumerate(parts):
        if i > 0:
            result.append(", ")
        if "×" in part:
            color_part, count_part = part.split("×", 1)
            result.append(color_gem_name(color_part))
            result.append(f"×{count_part}")
        else:
            result.append(part)
    return result
