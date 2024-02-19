import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy.typing import NDArray
from qc_utils.gates import weyl_decompose

def plot_weyl_traj(unitaries: NDArray[np.complex_], savepath: str | None = None) -> None:
    """Plot the Weyl coordinates of a sequence of unitaries.
    TODO: allow existing matplotlib axis to be passed as argument, which
    will then be drawn onto.

    Args:
        unitaries: NxMxM array of unitaries, where N is number of samples and M
            is Hilbert space dimension.
        savepath: (optional) filepath to save figure.
    """
    if unitaries.shape[0] > 100:
        step = unitaries.shape[0] // 100 + 1
    else:
        step = 1
    unitaries = unitaries[np.arange(0, unitaries.shape[0], step)]

    fig = plt.figure(figsize=(6,4))
    ax0 = fig.add_subplot(projection='3d')

    coords = []
    counter = 0
    for i,u in enumerate(unitaries):
        a,b,c,_,_,_,_,_ = weyl_decompose(u)
        coords.append([a,b,c])
        counter += 1
    coords = np.array(coords)
    colors = np.linspace(0,1,counter)

    colors = ax0.scatter(coords[:,0], coords[:,1], coords[:,2], c=colors, cmap='plasma')

    ax0.plot([0,1],[0,0],[0,0], c='black', alpha=0.3)
    ax0.plot([0,1/2],[0,1/2],[0,0], c='black', alpha=0.3)
    ax0.plot([0,1/2],[0,1/2],[0,1/2], c='black', alpha=0.3)
    ax0.plot([1,1/2],[0,1/2],[0,0], c='black', alpha=0.3)
    ax0.plot([1,1/2],[0,1/2],[0,1/2], c='black', alpha=0.3)
    ax0.plot([1/2,1/2],[1/2,1/2],[0,1/2], c='black', alpha=0.3)

    ax0.set_box_aspect((2,1,1))
    ax0.view_init(elev=30, azim=-70)
    cbar = plt.colorbar(colors, label='Time')
    cbar.draw_all()
    ax0.grid(False)
    plt.title('Weyl trajectory')
    # ax0.set_xticks([])
    # ax0.set_yticks([])
    # ax0.set_zticks([])
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath + 'points.svg')
        plt.savefig(savepath + 'points.pdf')
        plt.savefig(savepath + 'points.png')
    plt.show()

def add_cbar(
        ax: plt.Axes, 
        norm: mpl.colors.Normalize, 
        cmap: mpl.colors.Colormap | str, 
        loc: str = 'right',
        size: float = 0.04, 
        pad: float = 0.05,
        extend_figure: bool = True,
    ) -> mpl.colorbar.Colorbar:
    """Add colorbar to existing axis.

    Args:
        ax: Matplotlib axis to add colorbar to.
        norm: Matplotlib norm object.
        cmap: Matplotlib colormap object.
        loc: Location of colorbar.
        size: Size of colorbar.
        pad: Padding between colorbar and axis.
        extend_figure: If True, extend figure to fit colorbar. If False,
            the figure will be shrunk to fit the colorbar.
    """
    fig = ax.get_figure()
    assert fig is not None
    if extend_figure:
        position = ax.get_position()
        cbar_ax = fig.add_axes([position.x1 + pad/5, position.y0, size, position.y1 - position.y0])
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    else:
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes(loc, size=size*5, pad=pad)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    return cbar