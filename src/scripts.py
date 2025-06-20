import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib import legend_handler
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator
import numpy as np

# import sys
# sys.path.append('../buqeyenm/nuclear-matter-convergence/')  # decouple
# from nuclear_matter.graphs import *
# from nuclear_matter import fermi_momentum


def setup_rc_params(presentation=False):
    """Set matplotlib's rc parameters for the plots
        Parameters
        ----------
        presentation : boolean
            increases font size (more readable) for talks if enabled
    """
    if presentation:
        fontsize = 11
    else:
        fontsize = 9
    black = 'k'

    mpl.rcdefaults()  # Set to defaults

    mpl.rc('text', usetex=True)
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'

    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['axes.edgecolor'] = black
    # mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams['axes.labelcolor'] = black
    mpl.rcParams['axes.titlesize'] = fontsize

    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['xtick.color'] = black
    mpl.rcParams['ytick.color'] = black
    # Make the ticks thin enough to not be visible at the limits of the plot (over the axes border)
    mpl.rcParams['xtick.major.width'] = mpl.rcParams['axes.linewidth'] * 0.95
    mpl.rcParams['ytick.major.width'] = mpl.rcParams['axes.linewidth'] * 0.95
    # The minor ticks are little too small, make them both bigger.
    mpl.rcParams['xtick.minor.size'] = 2.4  # Default 2.0
    mpl.rcParams['ytick.minor.size'] = 2.4
    mpl.rcParams['xtick.major.size'] = 3.9  # Default 3.5
    mpl.rcParams['ytick.major.size'] = 3.9

    ppi = 72  # points per inch
    # dpi = 150
    mpl.rcParams['figure.titlesize'] = fontsize
    mpl.rcParams['figure.dpi'] = 150  # To show up reasonably in notebooks
    mpl.rcParams['figure.constrained_layout.use'] = True
    # 0.02 and 3 points are the defaults:
    # can be changed on a plot-by-plot basis using fig.set_constrained_layout_pads()
    mpl.rcParams['figure.constrained_layout.wspace'] = 0.0
    mpl.rcParams['figure.constrained_layout.hspace'] = 0.0
    mpl.rcParams['figure.constrained_layout.h_pad'] = 3. / ppi  # 3 points
    mpl.rcParams['figure.constrained_layout.w_pad'] = 3. / ppi

    mpl.rcParams['legend.title_fontsize'] = fontsize
    mpl.rcParams['legend.fontsize'] = fontsize
    mpl.rcParams['legend.edgecolor'] = 'inherit'  # inherits from axes.edgecolor, to match
    mpl.rcParams['legend.facecolor'] = (1, 1, 1, 0.6)  # Set facecolor with its own alpha, so edgecolor is unaffected
    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['legend.borderaxespad'] = 0.8
    mpl.rcParams['legend.framealpha'] = None  # Do not set overall alpha (affects edgecolor). Handled by facecolor above
    mpl.rcParams['patch.linewidth'] = 0.8  # This is for legend edgewidth, since it does not have its own option

    mpl.rcParams['hatch.linewidth'] = 0.5

    # bbox = 'tight' can distort the figure size when saved (that's its purpose).
    # mpl.rc('savefig', transparent=False, bbox='tight', pad_inches=0.04, dpi=350, format='png')
    #mpl.rc('savefig', transparent=False, bbox=None, dpi=400, format='png')


def fermi_momentum(density, degeneracy):
    """Computes the Fermi momentum of infinite matter in inverse fermi

    Parameters
    ----------
    density : array
        The density in inverse fermi^3
    degeneracy : int
        The degeneracy factor [g]. Equals 2 for neutron matter and 4 for symmetric matter.
    """
    return (6 * np.pi**2 * density / degeneracy)**(1./3)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def highlight_nsat(ax, nsat=0.164, zorder=0, band=False):
    ax.axvline(nsat, ls="--", lw=0.8, c='0.1', zorder=zorder)
    if band:
        import matplotlib.transforms as transforms
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        n0_std = 0.007
        rect = mpatches.Rectangle(
            (nsat-n0_std, 0), width=2*n0_std, height=1,
            transform=trans, facecolor='0.85', edgecolor='0.6',
            linewidth=0.6,
            alpha=0.7, zorder=zorder-0.01
        )
        ax.add_patch(rect)

def curve_plus_bands_plot(ax, x, y, std, color_68=None, color_95=None, 
                          zorder=None, zorder_c=None, edgecolor=None, **kwargs):
    """
    Plot y vs. x with one sigma and two sigma bands based on std on axis ax.
     Add any other keyword pairs to style the main curve.
    """
    ax.plot(x, y, zorder=zorder_c, **kwargs)
    if color_95 is not None:
        ax.fill_between(x, y + 2*std, y - 2*std, facecolor=color_95, 
                        edgecolor=edgecolor, zorder=zorder)
    if color_68 is not None:
        ax.fill_between(x, y + std, y - std, facecolor=color_68, 
                        edgecolor=edgecolor, zorder=zorder)
    return ax

def plot_obs_vs_density(
    density, y, std, density_data=None, y_data=None, add_nsat=True, ax=None, c='k',
    color_68=None, color_95=None, markersize=3, edgecolor=None, zorder=0, zorder_c =0, density_label=None, wrt_kf=False,
    **kwargs
):
    if wrt_kf is True:
        kf = fermi_momentum(density,4)
    if ax is None:
        fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(7, 3))
    curve_plus_bands_plot(
        ax, density, y, std, c=c, color_68=color_68, color_95=color_95,
        edgecolor=edgecolor, zorder=zorder, zorder_c=zorder_c, **kwargs
    )
    ax.set_xlabel(density_label)
    ax.margins(x=0)
    
    if y_data is not None:
        c_marker = c if c is not None else 'k'
        ax.plot(
            density_data, y_data, ls='', marker='o', c=c_marker, markersize=markersize,
            zorder=zorder, **kwargs
        )
        ax.plot(
            density_data, y_data, ls='', marker='o', c=c_marker, markersize=markersize,
            zorder=zorder, **kwargs
        )
    if add_nsat:
        highlight_nsat(ax, zorder=zorder+10)

    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    
    return ax

def plot_obs_panels(density, y, dy, orders, density_data=None, y_data=None, colors=None, axes=None, density_label=None, \
                    order_labels=None, **kwargs):
    
    if axes is None:
        fig, axes = plt.subplots(2, 2, sharey='row', sharex='col', figsize=(3.4, 3.4))
    fig = plt.gcf()
    fig.set_constrained_layout_pads(h_pad=1.1/72, w_pad=1.2/72)
    
    if colors is None:
        color_list = ['Oranges', 'Greens', 'Blues', 'Reds', 'Purples', 'Greys']
        cmaps = [plt.get_cmap(name) for name in color_list[:len(orders)]]
        colors = [cmap(0.55 - 0.1 * (i == 0)) for i, cmap in enumerate(cmaps)]
    
    light_colors = [lighten_color(color) for color in colors]
    dark_colors = [lighten_color(color, 1.5) for color in colors]

    for j, ax in enumerate(axes.ravel()):
        for i, n in enumerate(orders[:j+1]):
            if y_data is not None:
                y_data_i = y_data[:, i]
            else:
                y_data_i = None

            plot_obs_vs_density(
                density,
                y[:, i],
                dy[:, i],
                ax=ax,
#                 color_68=colors[i],
#                 color_95=light_colors[i],
#                 c=dark_colors[i],
                c=colors[i],
                color_68=light_colors[i],
                edgecolor=colors[i],
                add_nsat=i==j,
                zorder=i/3,
                zorder_c=i/3,
                density_data=density_data,
                y_data=y_data_i,
                markersize=3,
                **kwargs
            )
        ax.axhline(0, 0, 1, c='k', lw=0.8)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(right=True, top=True, which='both')
    add_top_order_legend(fig, axes[0, 0], axes[0, 1], order_labels, colors, light_colors, dark_colors)
    axes[0,0].set_xlabel('')
    axes[0,1].set_xlabel('')
    return fig, axes

class OrderBandsHandler(legend_handler.HandlerPolyCollection):

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        light_rect = orig_handle[0]
        dark_rect = orig_handle[1]
        dark_line = orig_handle[2]
        dark_color = dark_line.get_color()
        color = dark_rect.get_facecolor()[0]
        light_color = light_rect.get_facecolor()[0]
        lw = 0.8

        outer_rect = mpatches.Rectangle(
            [0, 0], 1, 1, facecolor='none', edgecolor=dark_color,
            lw=lw
        )
        dark_rect = mpatches.Rectangle(
            [0, 0], 1, 1, facecolor=color, edgecolor='none',
            lw=0
        )
        light_rect = mpatches.Rectangle(
            [0, 0], 1, 1, facecolor=light_color, edgecolor='none',
            lw=0
        )

        patches = []

        factor = 2
        dark_height = 0.4 * height
        #         patches += super().create_artists(legend, light_rect, xdescent, ydescent, width, height, fontsize, trans)
        #         patches += super().create_artists(
        #             legend, dark_rect, xdescent, ydescent-height/(2*factor), width, height/factor, fontsize, trans)
        # print(ydescent, height)
        patches += legend_handler.HandlerPatch().create_artists(
            legend, light_rect, xdescent, ydescent, width, height, fontsize, trans)
        # patches += legend_handler.HandlerPatch().create_artists(
        #     legend, dark_rect, xdescent, ydescent - height / (2 * factor), width, height / factor, fontsize, trans)
        patches += legend_handler.HandlerPatch().create_artists(
            legend, dark_rect, xdescent, ydescent - height / 2 + dark_height/2, width, dark_height, fontsize, trans)

        outer_patches = legend_handler.HandlerPatch().create_artists(
            legend, outer_rect, xdescent, ydescent, width, height, fontsize, trans)
        outer_patches[0].set_linewidth(lw / 2)
        patches += outer_patches

        # line_patches = HandlerLine2D().create_artists(
        #     legend, dark_line, xdescent, ydescent, width, height, fontsize, trans)
        # line_patches[0].set_linewidth(lw)
        # patches += line_patches

        return patches


def compute_filled_handles(colors, light_colors, dark_colors):
    handles = []
    for color, light_color, dark_color in zip(colors, light_colors, dark_colors):
        fill_light = plt.fill_between([], [], [], alpha=1, color=light_color)
        fill_normal = plt.fill_between([], [], [], alpha=1, color=color)
        line_dark = plt.plot([], [], color=dark_color, dash_capstyle='butt', solid_capstyle='butt')[0]

        handles.append((fill_light, fill_normal, line_dark))
    return handles


def add_top_order_legend(fig, ax_left, ax_right, orders, colors, light_colors, dark_colors):
    fig.canvas.draw()  # Must draw to get positions right before getting locations
    # Get the corner of the upper right plot in display coordinates
    upper_right_display = ax_right.transAxes.transform((1, 1))
    # Now put it in axes[0,0] coords
    upper_right_axes00 = ax_left.transAxes.inverted().transform(upper_right_display)
    handlers_ords = compute_filled_handles(colors, light_colors, dark_colors)
    # Must use axes[0,0] legend for constrained layout to work with it
    return ax_left.legend(
        handlers_ords, orders, ncol=4,
        loc='lower left', bbox_to_anchor=(0, 1.02, upper_right_axes00[0], 1.),
        mode='expand',
        columnspacing=0,
        handletextpad=0.5,
        fancybox=False, borderaxespad=0,
        handler_map={tuple: OrderBandsHandler()}
    )
