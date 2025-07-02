import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def configure_matplotlib():
    sns.set()
    sns.set_style(style="whitegrid")
    sns.set_palette("colorblind", 6)
    sns.set_context("paper")
    matplotlib.rcParams["ps.useafm"] = True
    matplotlib.rcParams["pdf.use14corefonts"] = True
    matplotlib.rcParams["legend.columnspacing"] = 1.0
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.major.pad"] = "0"
    plt.rcParams["ytick.major.pad"] = "0"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({"font.size": 10})
    plt.rcParams.update({"axes.labelsize": 10})
    matplotlib.rcParams["svg.fonttype"] = "none"
    # default xlim[]
    plt.rcParams["axes.xmargin"] = 0


def set_one_column(hegiht=3.5):
    configure_matplotlib()
    plt.rcParams.update({"figure.figsize": (3.5, hegiht)})


def set_two_column(hegiht=3.5):
    configure_matplotlib()
    plt.rcParams.update({"figure.figsize": (7, hegiht)})
