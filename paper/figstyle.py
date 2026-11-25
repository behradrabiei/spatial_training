"""Shared format of the paper figures for a two-column layout (RSS / IEEE-style).

Figures are drawn at their final printed size, so the text is 7-8 pt on the page and they are
included without scaling: COLUMN_WIDTH wide for one column (\\columnwidth), TEXT_WIDTH for a
figure* spanning both. Plain matplotlib defaults otherwise. Enter COLUMN_RC with
``plt.rc_context`` around both drawing and saving (pdf.fonttype is read at save time).

A plot script imports it with ``sys.path.insert(0, str(HERE.parent))``.
"""
COLUMN_WIDTH = 3.5
TEXT_WIDTH = 7.16
COLUMN_RC = {"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
             "ytick.labelsize": 7, "legend.fontsize": 7, "lines.linewidth": 1.2, "axes.linewidth": 0.6,
             "grid.linewidth": 0.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
             "xtick.major.size": 2.5, "ytick.major.size": 2.5,
             "pdf.fonttype": 42, "ps.fonttype": 42}  # TrueType in the PDF, no Type 3 fonts


def top_legend(fig, handles, labels):
    """Legend centred above the panels, in as few rows as fit the figure width. Returns the legend
    height as a figure fraction (pass ``1 - height`` as the top of ``tight_layout``'s rect)."""
    renderer = fig.canvas.get_renderer()
    for rows in range(1, len(labels) + 1):
        legend = fig.legend(handles, labels, loc="upper center", ncol=-(-len(labels) // rows), frameon=False,
                            bbox_to_anchor=(0.5, 1.0), handlelength=2.8, columnspacing=1.0,
                            handletextpad=0.5, borderaxespad=0.2)
        box = legend.get_window_extent(renderer)
        if box.width <= fig.bbox.width or rows == len(labels):
            return (fig.bbox.height - box.y0) / fig.bbox.height
        legend.remove()
