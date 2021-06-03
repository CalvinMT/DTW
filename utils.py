# ---------- --COLOUR-- ----------
RESET_COLOR_ESCAPE = '\033[0m'

def get_color_escape(r, g, b, background=False):
    return '\033[{};2;{};{};{}m'.format(48 if background else 38, r, g, b)

def hexToRGB(hex):
    h = hex.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
# ---------- ---------- ----------
