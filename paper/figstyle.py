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
