import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional

# from paul tol
tol_colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']

# from https://www.nature.com/articles/nmeth.1618
nature_colors = ['#0072B2', '#CC79A7', '#009E73', '#E69F00', '#56B4E9', '#D55E00', '#F0E442']

opensans_path = '/Users/jchad/Library/Fonts/OpenSans-VariableFont_wdth,wght.ttf'

def set_font(path: str = opensans_path, size: Optional[int] = 12) -> None:
    """Set default matplotlib font.

    Args:
        path: file path to font file.
        size: font size.
    """    

    mpl.font_manager.fontManager.addfont(path)
    prop = mpl.font_manager.FontProperties(fname=path)

    plt.rcParams['font.family'] = 'fantasy'
    plt.rcParams['font.fantasy'] = prop.get_name()
    
    if size is not None:
        plt.rcParams['font.size'] = size

def set_colors(colors: list[str] = nature_colors) -> None:
    """Set matplotlib default colors to `colors`.

    Args:
        colors: list of hexadecimal color codes, e.g. "#0072B2".
    """
    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) 

def setup_default() -> None:
    """Call set_font and set_colors with default options.
    """
    set_font()
    set_colors()