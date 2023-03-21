from matplotlib import font_manager
import matplotlib.pyplot as plt

colors = ['cornflowerblue', 'firebrick', 'forestgreen', 'orchid', 'sandybrown']

def set_font(font='Open Sans', path=None):
    font_path = None
    if path is None:
        if font == 'Open Sans':
            font_path = '/usr/share/fonts/truetype/OpenSans/OpenSans-VariableFont_wdth,wght.ttf'  # Your font path goes here
    if font_path is None:
        plt.rcParams['font.family'] = font_path
    else:
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)

        plt.rcParams['font.family'] = 'fantasy'
        plt.rcParams['font.fantasy'] = prop.get_name()