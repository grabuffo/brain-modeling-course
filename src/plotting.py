"""Shared figure style for the course notebooks."""
import matplotlib.pyplot as plt
import seaborn as sns

COLORS = sns.color_palette("muted")


def setup():
    """Call once at the top of every notebook."""
    sns.set_theme(context="notebook", style="ticks", palette="muted")
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
    })


def slider(value, lo, hi, step, description):
    """A slider that only updates the figure when you release it (keeps Colab responsive)."""
    from ipywidgets import FloatSlider, Layout
    return FloatSlider(value=value, min=lo, max=hi, step=step, description=description,
                       continuous_update=False, style={"description_width": "initial"},
                       layout=Layout(width="420px"))


def phase_movie(t, phases, Z, n_show=200, stride=2, interval=50, title="", spike_phase=None):
    """Animate units moving on the unit circle, with the order parameter Z = R e^{i Psi} as an arrow.

    t: times (n_t,), phases: (n_t, n_units) unwrapped phases, Z: (n_t,) complex.
    Left: the circle. Right: R(t) with a moving cursor.
    Returns a matplotlib FuncAnimation; show it with show_movie(anim).
    """
    import numpy as np
    from matplotlib import animation
    idx = np.arange(0, len(t), stride)
    ph = np.mod(phases[:, :n_show], 2 * np.pi)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw=dict(width_ratios=[1, 1.4]))
    c = np.linspace(0, 2 * np.pi, 200)
    ax[0].plot(np.cos(c), np.sin(c), c="lightgray", lw=1.5)
    if spike_phase is not None:
        ax[0].plot(np.cos(spike_phase), np.sin(spike_phase), "*", c="C3", ms=14)
    dots = ax[0].scatter(np.cos(ph[0]), np.sin(ph[0]), s=12, color=COLORS[0], alpha=0.7)
    arrow = ax[0].annotate("", xy=(Z[0].real, Z[0].imag), xytext=(0, 0),
                           arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
    ax[0].set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal"); ax[0].axis("off")
    ax[0].set_title(title, fontsize=10)
    ax[1].plot(t, np.abs(Z), "k", lw=0.8)
    cursor = ax[1].axvline(t[0], c="C3")
    ax[1].set(xlabel="time", ylabel="R = |Z|", ylim=(0, 1.05), title="order parameter")
    plt.tight_layout()

    def update(k):
        i = idx[k]
        dots.set_offsets(np.column_stack([np.cos(ph[i]), np.sin(ph[i])]))
        arrow.xy = (Z[i].real, Z[i].imag)
        cursor.set_xdata([t[i], t[i]])
        return dots, arrow, cursor

    anim = animation.FuncAnimation(fig, update, frames=len(idx), interval=interval, blit=False)
    plt.close(fig)
    return anim


def show_movie(anim, save_as=None):
    """Display an animation in the notebook (mp4 if ffmpeg is available, else JavaScript player)."""
    from IPython.display import HTML
    from matplotlib import animation
    if save_as is not None and animation.writers.is_available("ffmpeg"):
        anim.save(save_as, writer="ffmpeg", dpi=110)
    if animation.writers.is_available("ffmpeg"):
        return HTML(anim.to_html5_video())
    return HTML(anim.to_jshtml())
