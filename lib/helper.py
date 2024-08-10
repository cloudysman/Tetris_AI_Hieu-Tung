from typing import List, Any

def grid_to_str(grid: List[List[Any]]) -> str:
    return "\n".join(" ".join(str(sq) for sq in row) for row in grid)

def copy_2d(grid: List[List[Any]]) -> List[List[Any]]:
    return [list(row) for row in grid]

def text_list_flatten(text_list: List[str]) -> str:
    return ", ".join(s for s in text_list if isinstance(s, str))