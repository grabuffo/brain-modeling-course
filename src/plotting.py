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


def sheet_movie(t, phases, rates, titles, stride=1, interval=60):
    """Animate one or more rotator sheets side by side (see networks.rotator_sheet).

    phases: list of arrays (n_t, L, L); rates: list of arrays (n_t,); titles: list of str.
    Top: each sheet coloured by sin(phi) (bright = the unit is spiking, dark = at rest).
    Bottom: the firing rate of the whole sheet, with a moving cursor.
    """
    import numpy as np
    from matplotlib import animation
    n = len(phases)
    idx = np.arange(0, len(t), stride)
    fig, ax = plt.subplots(2, n, figsize=(4.2 * n, 5.6), gridspec_kw=dict(height_ratios=[3, 1.1]), squeeze=False)
    ims, cursors = [], []
    for j in range(n):
        im = ax[0, j].imshow(np.sin(phases[j][0]), cmap="magma", vmin=-1, vmax=1, interpolation="nearest")
        ax[0, j].set_title(titles[j], fontsize=10); ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
        ax[1, j].plot(t, rates[j], "k", lw=0.8)
        ax[1, j].set(xlabel="time", ylabel="rate of the sheet", xlim=(t[0], t[-1]))
        cursors.append(ax[1, j].axvline(t[0], c="C3")); ims.append(im)
    cb = fig.colorbar(ims[-1], ax=ax[0, :].tolist(), shrink=0.8, pad=0.02)
    cb.set_label(r"sin $\varphi$  (bright = spike)")

    def update(k):
        i = idx[k]
        for j in range(n):
            ims[j].set_data(np.sin(phases[j][i])); cursors[j].set_xdata([t[i], t[i]])
        return ims + cursors

    anim = animation.FuncAnimation(fig, update, frames=len(idx), interval=interval, blit=False)
    plt.close(fig)
    return anim


def sheet_playground():
    """A live, interactive rotator sheet that runs in the browser (same equations as networks.rotator_sheet,
    written in JavaScript so that it can run in real time while you move the sliders)."""
    import uuid
    from pathlib import Path
    from IPython.display import HTML
    html = (Path(__file__).with_name("sheet_playground.html")).read_text()
    return HTML(html.replace("UID", "sheet-" + uuid.uuid4().hex[:8]))
