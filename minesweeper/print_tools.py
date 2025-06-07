from enum import Enum

COL_MAP = {
    "black":   [0, 0, 0],
    "red":     [255, 0, 0],
    "green":   [0, 230, 0],
    "yellow":  [240, 240, 0],
    "blue":    [20, 138, 255],
    "magenta": [255, 0, 255],
    "cyan":    [0, 255, 255],
    "white":   [255, 255, 255],
    "gray":    [128, 128, 128],
    # "pink":     [255, 0, 128],
    "orange":   [255, 128, 0],
    "violet":   [128, 0, 255],
}

TXT_COL = {
    '0': None,
    '1': COL_MAP["blue"],
    '2': COL_MAP["green"],
    '3': COL_MAP["orange"],
    '4': COL_MAP["violet"],
    '5': COL_MAP["cyan"],
    '6': COL_MAP["gray"],
    '7': COL_MAP["yellow"],
    '8': COL_MAP["white"],
    'X': COL_MAP["red"],
    '#': None,
}

class PrintMode(Enum):
    NoPrint = -1
    Normal = 0
    RevealMines = 1
    RevealAll = 2

def print_styled(text: str, bold: bool = False, fg_rgb: list = None, bg_rgb: list = None, doPrint = False):
    """
    Print text with optional bold style and optional RGB foreground/background colors.

    Args:
        text (str): The text to print.
        bold (bool): Whether the text should be bold. Defaults to False.
        fg_rgb (list or None): Foreground RGB color as a list [r, g, b], or None for default color.
        bg_rgb (list or None): Background RGB color as a list [r, g, b], or None for default background.
    """
    codes = []

    # Bold or normal intensity
    codes.append("1" if bold else "22")

    # Foreground color
    if fg_rgb is not None:
        if len(fg_rgb) != 3:
            raise ValueError("Foreground color must be a list of 3 integers.")
        codes.append(f"38;2;{fg_rgb[0]};{fg_rgb[1]};{fg_rgb[2]}")

    # Background color
    if bg_rgb is not None:
        if len(bg_rgb) != 3:
            raise ValueError("Background color must be a list of 3 integers.")
        codes.append(f"48;2;{bg_rgb[0]};{bg_rgb[1]};{bg_rgb[2]}")

    # Build and print the escape sequence
    escape_seq = f"\033[{';'.join(codes)}m{text}\033[0m"

    if doPrint:
        print(escape_seq)
    return escape_seq


if __name__ == "__main__":
    for c in COL_MAP:
        print_styled("abc",bold=True,fg_rgb=COL_MAP[c])


