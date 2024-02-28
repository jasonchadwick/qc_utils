import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy.typing import NDArray
import qutip as qt
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

def plot_state_evolutions(
        tlist: NDArray[np.float_],
        states: list[qt.Qobj],
        target_states: list[qt.Qobj] | None = None,
        target_state_labels: list[str] | None = None,
        levels_to_keep: list[int] | None = None,
        nt: list[int] | None = None,
        population_cutoff_threshold: float = 1e-3,
        ax: plt.Axes | None = None,
    ):
    """Plot the evolution of quantum states over time.

    Args:
        tlist: Array of times.
        states: List of quantum states at each time.
        target_states: Array of target quantum states to compare against and
            show overlap. If None, use either levels_to_keep or use standard
            basis.
        target_state_labels: Labels for the target states.
        levels_to_keep: Number of energy levels per subsystem to include in the
            plot. For example, if nt = [3,3] (two qutrits) and levels_to_keep =
            [2,2], the 4 qubit states will be plotted.
        nt: Array of dimensions of the quantum states. Required if 
            target_states is None.
        population_cutoff_threshold: State overlaps are only plotted if their  
            maximum population is above this threshold.

    Returns:
        Matplotlib axis.
    """
    if target_states is None:
        target_states = []
        target_state_labels = []
        if nt is None:
            raise ValueError('nt must be provided if target_states is None')
        if levels_to_keep is None:
            levels_to_keep = nt
        if np.any(levels_to_keep > nt):
            raise ValueError(f'levels_to_keep > nt')
        for state in product(*[range(n) for n in levels_to_keep]):
            target_states.append(qt.basis(nt, list(state)))
            target_state_labels.append(f'{state}')
    assert target_states is not None
    assert target_state_labels is not None

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    assert ax is not None

    for i,target_state in enumerate(target_states):
        probs = [np.abs((target_state.dag() * state).full()[0,0])**2 for state in states]
        if np.max(probs) > population_cutoff_threshold:
            ax.plot(tlist, probs, label=f'{target_state_labels[i]}')
    ax.legend()
    return ax

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