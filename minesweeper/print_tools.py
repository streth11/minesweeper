from enum import Enum

def ansi_demo():
    foregrounds = {
        "Black": 30, "Red": 31, "Green": 32, "Yellow": 33,
        "Blue": 34, "Magenta": 35, "Cyan": 36, "White": 37,
        "Bright Black": 90, "Bright Red": 91, "Bright Green": 92, "Bright Yellow": 93,
        "Bright Blue": 94, "Bright Magenta": 95, "Bright Cyan": 96, "Bright White": 97
    }

    backgrounds = {
        "Black": 40, "Red": 41, "Green": 42, "Yellow": 43,
        "Blue": 44, "Magenta": 45, "Cyan": 46, "White": 47,
        "Bright Black": 100, "Bright Red": 101, "Bright Green": 102, "Bright Yellow": 103,
        "Bright Blue": 104, "Bright Magenta": 105, "Bright Cyan": 106, "Bright White": 107
    }

    print("ANSI Color Chart (FG on BG):\n")
    for bg_name, bg_code in backgrounds.items():
        print(f" BG: {bg_name.ljust(15)} ", end="")
        for fg_name, fg_code in list(foregrounds.items()):  # use only base 8 FG for readability
            print(f"\033[1;{fg_code};{bg_code}m {fg_name[0]} \033[0m", end=" ")
        print()

# ansi_demo()

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
    Normal = 0
    RevealMines = 1
    RevealAll = 2

def print_styled(text: str, bold: bool = False, fg_rgb: list = None, bg_rgb: list = None):
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
    print(escape_seq)

for c in COL_MAP:

    print_styled("abc",bold=True,fg_rgb=COL_MAP[c])


