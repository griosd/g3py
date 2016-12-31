import matplotlib.pyplot as plt
import seaborn as sb
import IPython.display as display
from g3py import config

def style_seaborn():
    sb.set(style='darkgrid', color_codes=False)
    plt.rcParams['figure.figsize'] = (20, 6)  #figure size
    config.plot_big = False


def style_normal():
    sb.set(style="white", color_codes=True) #white background
    plt.rcParams['figure.figsize'] = (20, 6)  #figure size
    plt.rcParams['axes.titlesize'] = 20  # title size
    plt.rcParams['axes.labelsize'] = 18  # xy-label size
    plt.rcParams['xtick.labelsize'] = 16 #x-numbers size
    plt.rcParams['ytick.labelsize'] = 16 #y-numbers size
    plt.rcParams['legend.fontsize'] = 18  # legend size
    #plt.rcParams['legend.fancybox'] = True
    config.plot_big = False


def style_big():
    sb.set(style="white", color_codes=True) #white background
    plt.rcParams['figure.figsize'] = (20, 6)  #figure size
    plt.rcParams['xtick.labelsize'] = 36  # x-numbers size
    plt.rcParams['ytick.labelsize'] = 36  # x-numbers size
    plt.rcParams['axes.labelsize'] = 36  # xy-label size
    plt.rcParams['axes.titlesize'] = 36  # xy-label size
    plt.rcParams['legend.fontsize'] = 20  # legend size
    config.plot_big = True


def style_big_seaborn():
    style_seaborn()
    plt.rcParams['xtick.labelsize'] = 36  # x-numbers size
    plt.rcParams['ytick.labelsize'] = 36  # x-numbers size
    plt.rcParams['axes.labelsize'] = 36  # xy-label size
    plt.rcParams['axes.titlesize'] = 36  # xy-label size
    plt.rcParams['legend.fontsize'] = 30  # legend size
    config.plot_big = True


def style_widget():
    return display.display(display.HTML('''<style>
                .widget-label { min-width: 30ex !important; }
                .widget-hslider { min-width:100%}
            </style>'''))


def plot(*args, **kwargs):
    plt.plot(*args, **kwargs)


def show(*args, **kwargs):
    plt.show(*args, **kwargs)


def plot_text(title="title", x="xlabel", y="ylabel", ncol=3, loc=8, axis=None):
    plt.axis('tight')
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(ncol=ncol, loc=loc)
    if axis is not None:
        plt.axis(axis)


def save_plot(file='example.pdf'):
    plt.savefig(file, bbox_inches='tight')