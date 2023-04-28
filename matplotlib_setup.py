from matplotlib import font_manager

import matplotlib as mpl
import matplotlib.pyplot as plt

# colors = ['cornflowerblue', 'firebrick', 'forestgreen', 'orchid', 'sandybrown']
colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']

def set_font(font='Open Sans', path=None, size=None):
    font_path = None
    if path is None:
        if font == 'Open Sans':
            font_path = '/Users/jchad/Library/Fonts/OpenSans-VariableFont_wdth,wght.ttf'  # Your font path goes here
    if font_path is None:
        plt.rcParams['font.family'] = font_path
    else:
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)

        plt.rcParams['font.family'] = 'fantasy'
        plt.rcParams['font.fantasy'] = prop.get_name()
    
    if size is not None:
        plt.rcParams['font.size'] = size

def set_colors(c=colors):
    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=c) 

def setup_default():
    set_font()
    set_colors()