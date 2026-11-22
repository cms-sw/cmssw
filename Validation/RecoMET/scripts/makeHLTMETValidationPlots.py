#!/usr/bin/env python3

import os
import re
import argparse
import numpy as np
import hist
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.colors import LogNorm
plt.style.use('tableau-colorblind10')
import array
import ROOT
import mplhep as hep
hep.style.use("CMS")
from dataclasses import dataclass

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

@dataclass
class HLTMETInputs:
    x: str
    y: str
    hist: str = None     # name of the ROOT histogram; defaults to the key under which the inputs are stored
                         # (which also names the output plot), so that one histogram can be plotted several times
    rebinx: int = None   # x-axis rebinning (1D and 2D)
    rebiny: int = None   # y-axis rebinning (2D only)
    out: str = ''
    logy: bool = False
    logz: bool = False
    drawdiag: bool = False
    
# Axis ranges per sample, to restrict the plotted range if needed (see --sample and --axis_limits):
#   {sample: {plot name: {'x': (low, high), 'y': (low, high)}}}
# The plot name is the output name (e.g. 'MET_logy'), or else the histogram name (e.g. 'MET'); each axis is optional.
# Plots not listed keep the full range, i.e. behave as with --no-axis_limits.
# All plots with the number of vertices share the same range, whatever the sample.
NVTX_LIMITS = {
    'Nvertex'                : {'x': (0, 350)},
    'MET_Nvtx'               : {'x': (0, 350)},
    'METRespAggr_Nvtx'       : {'x': (0, 350)},
    'METDiffAggr_Nvtx'       : {'x': (0, 350)},
    'METDiffResolAggr_Nvtx'  : {'x': (0, 350)},
    'METxDiffAggr_Nvtx'      : {'x': (0, 350)},
    'METxDiffResolAggr_Nvtx' : {'x': (0, 350)},
    'METyDiffAggr_Nvtx'      : {'x': (0, 350)},
    'METyDiffResolAggr_Nvtx' : {'x': (0, 350)},
    'METRatiovsNvtx'         : {'y': (0, 350)},
    'METDiffvsNvtx'          : {'y': (0, 350)},
    'METxDiffvsNvtx'         : {'y': (0, 350)},
    'METyDiffvsNvtx'         : {'y': (0, 350)},
}
AXIS_LIMITS = {
    'QCD': {
        'MET'     : {'x': (0, 800)},
        'MET_logy': {'x': (0, 800)},
        **NVTX_LIMITS,
    },
    'Wprime': {
        **NVTX_LIMITS,
    },
}
DEFAULT_SAMPLE = 'QCD'

# DQM directory of the MET validation, per validation step (see --step)
DQM_DIRS = {
    'HLT': "DQMData/Run 1/HLT/Run summary/JetMET/METValidation",
    'Offline': "DQMData/Run 1/JetMET/Run summary/METValidation",
}

def resolveSample(sample, label):
    """
    Sample whose axis limits are used: the one given with --sample, otherwise the one named in the sample label
    (case-insensitive), otherwise DEFAULT_SAMPLE. Raises if both are given and refer to different samples.
    Returns the sample and a short explanation of the choice.
    """
    named = [k for k in AXIS_LIMITS if k.lower() in label.lower()]
    if len(named) > 1:
        raise ValueError(f"sample label '{label}' matches several samples: {', '.join(named)}")
    if sample is not None:
        if named and named[0] != sample:
            raise ValueError(f"--sample {sample} is incompatible with the sample label '{label}' (looks like {named[0]})")
        return sample, 'from --sample'
    if named:
        return named[0], f"inferred from the label '{label}'"
    return DEFAULT_SAMPLE, f"default, no sample found in the label '{label}'"

def printOptions(args, sample_why):
    """Prints all the options in use, one per line."""
    onoff = lambda flag: 'on' if flag else 'off'
    if args.compare_collections:
        mode, what = 'compare collections (same file)', ' '.join(args.compare_collections)
    elif args.compare_files:
        mode, what = 'compare files (same collection)', f"{args.met[0]} in " + ' | '.join(
            f"{f} [{l}]" for f, l in zip(args.compare_files, args.compare_files_labels))
    else:
        mode, what = 'single collections', ' '.join(args.met)
    if not args.axis_limits:
        limits = 'off (full range)'
    else:
        sample_limits = AXIS_LIMITS.get(args.sample, {})
        groups = {}  # plots sharing the same limits are listed together
        for k, lim in sample_limits.items():
            groups.setdefault(', '.join(f"{ax}={v}" for ax, v in lim.items()), []).append(k)
        limits = 'on, ' + ('; '.join(f"{lims}: {', '.join(names)}" for lims, names in groups.items())
                           or 'no restricted plot for this sample')
    rows = (
        ('Step',                  args.step),
        ('Mode',                  mode),
        ('Collections',           what),
        ('Input file',            args.file if args.file else '-'),
        ('Output directory',      args.odir),
        ('Sample label',          args.sample_label),
        ('Sample type',           f"{args.sample} ({sample_why})"),
        ('Period',                f"{args.period} ({args.energy} TeV)"),
        ('Axis limits',           limits),
        ('Under/overflow counts', onoff(args.show_flow)),
        ('Overlay of cut scan',   onoff(args.overlay_scan)),
    )
    width = max(len(k) for k, _ in rows)
    print(" ### INFO: Options")
    for k, v in rows:
        print(f"   {k:<{width}} : {v}")

def axis_limits(name, hist=None):
    """Axis limits of the plot `name` (or of its histogram `hist`) for the sample chosen with --sample."""
    if not args.axis_limits:
        return {}
    sample = AXIS_LIMITS.get(args.sample, {})
    return sample.get(name, sample.get(hist, {}))

def visible(bin_centers, lims):
    """Mask of the bins inside the x limits, used to autoscale y only on what is shown."""
    if 'x' not in lims:
        return np.ones(len(bin_centers), dtype=bool)
    mask = (bin_centers >= lims['x'][0]) & (bin_centers <= lims['x'][1])
    return mask if mask.any() else np.ones(len(bin_centers), dtype=bool)

def createDir(adir):
    if not os.path.exists(adir):
        os.makedirs(adir)
    return adir

def checkRootDir(afile, adir):
    if not afile.Get(adir):
        raise RuntimeError(f"Directory '{adir}' not found in {afile}")

def checkRootFile(afile, hname, rebinx=None, rebiny=None):
    hist_orig = afile.Get(hname)
    if not hist_orig:
        raise RuntimeError(f"WARNING: Histogram {hname} not found.")

    hist = hist_orig.Clone(hname + "_clone")
    hist.SetDirectory(0) # detach from file

    if hist.GetDimension() == 2:
        if rebinx is not None or rebiny is not None:
            for r in (rebinx, rebiny):
                if r is not None and not isinstance(r, int):
                    raise ValueError(f"2D rebinning only supports integer grouping, got {type(r)}")
            nx = rebinx if rebinx is not None else 1
            ny = rebiny if rebiny is not None else 1
            hist = hist.Rebin2D(nx, ny, hname + "_rebin")
    elif rebinx is not None:
        if isinstance(rebinx, (int, float)):
            hist = hist.Rebin(int(rebinx), hname + "_rebin")
        elif hasattr(rebinx, '__iter__'):
            bin_edges_c = array.array('d', rebinx)
            hist = hist.Rebin(len(bin_edges_c) - 1, hname + "_rebin", bin_edges_c)
        else:
            raise ValueError(f"Unknown type for rebin: {type(rebinx)}")

    return hist

def define_bins(h):
    """
    Computes the number of bins, edges, centers and widths of a histogram.
    """
    N = h.GetNbinsX()
    edges = np.array([h.GetBinLowEdge(i+1) for i in range(N)])
    edges = np.append(edges, h.GetBinLowEdge(N+1))
    return N, edges, 0.5*(edges[:-1]+edges[1:]), np.diff(edges)
            
def histo_values_errors(h):
    N = h.GetNbinsX()
    values = np.array([h.GetBinContent(i+1) for i in range(N)])
    errors = np.array([h.GetBinError(i+1) for i in range(N)])
    return values, errors

def define_bins_2D(h):
    Nx = h.GetNbinsX()
    Ny = h.GetNbinsY()

    x_edges = np.array([h.GetXaxis().GetBinLowEdge(i+1) for i in range(Nx)])
    x_edges = np.append(x_edges, h.GetXaxis().GetBinUpEdge(Nx))

    y_edges = np.array([h.GetYaxis().GetBinLowEdge(j+1) for j in range(Ny)])
    y_edges = np.append(y_edges, h.GetYaxis().GetBinUpEdge(Ny))

    return Nx, Ny, x_edges, y_edges

def histo_values_2D(h, error=False):
    Nx = h.GetNbinsX()
    Ny = h.GetNbinsY()
    values = np.array([
        [h.GetBinContent(i+1, j+1) for i in range(Nx)]
        for j in range(Ny)
    ])
    return values

class Plotter:
    def __init__(self, label, en_com, period='Phase-2', fontsize=18, grid_color='grey'):
        self._fig, self._ax = plt.subplots(figsize=(10, 10))
        self.fontsize = fontsize

        cms_text = f' {period} Simulation Preliminary'
        lumi_text = f"{label} | {en_com} TeV"
        cms_text_dict = dict(ax=self._ax, fontsize=fontsize)
        if hasattr(hep.cms, 'lumitext'): # mplhep 0.4
            hep.cms.text(cms_text, **cms_text_dict)
            hep.cms.lumitext(lumi_text, **cms_text_dict)
        else:                            # mplhep >= 1.0
            hep.cms.text(cms_text, lumi=lumi_text, **cms_text_dict)

        if grid_color:
            self._ax.grid(which='major', color=grid_color)
        
        self.extensions = ('png', 'pdf')

    @property
    def fig(self):
        return self._fig

    @property
    def ax(self):
        return self._ax

    def labels(self, x, y, legend_title=None, legend_loc='upper right'):
        self._ax.set_xlabel(x)
        self._ax.set_ylabel(y)
        if legend_title is not None:
            self._ax.legend(title=legend_title,
                            title_fontsize=self.fontsize, fontsize=self.fontsize,
                            loc=legend_loc)

    def limits(self, x=None, y=None, logY=False, logX=False):
        if x:
            self._ax.set_xlim(x)
        if y:
            self._ax.set_ylim(y)
        if logX:
            self._ax.set_xscale('log')
        if logY:
            self._ax.set_yscale('log')

    def apply_limits(self, lims):
        """Apply the user-defined axis limits (see AXIS_LIMITS); they take precedence over the autoscaling."""
        if 'x' in lims:
            self._ax.set_xlim(lims['x'])
        if 'y' in lims:
            self._ax.set_ylim(lims['y'])

    def make_room_above(self, text, data_max, logY=False, margin=0.05):
        """
        Raises the top y limit, if needed, so that `text` lies entirely above the data:
        the highest data point plus a `margin` (fraction of the axis height) must stay below the text.
        """
        if not np.isfinite(data_max) or data_max <= 0.:
            return
        self._fig.tight_layout() # final layout, so that the text extent is measured in the right axes
        self._fig.canvas.draw()
        bottom = self._ax.transAxes.inverted().transform(text.get_window_extent())[:, 1].min() # axes fraction
        lo, hi = self._ax.get_ylim()
        if not 0. < bottom < 1.:
            return
        if logY:
            top = data_max * (hi / lo) ** margin
            needed = lo * (top / lo) ** (1. / bottom)
        else:
            top = data_max + margin * (hi - lo)
            needed = lo + (top - lo) / bottom
        if needed > hi:
            self._ax.set_ylim(lo, needed)

    def limits_log(self, pmin, amax):
        """y limits for a log scale: from half the smallest positive content to a factor above the maximum."""
        if np.isfinite(pmin) and amax > 0.:
            self.limits(y=(0.5*pmin, 5.*amax), logY=True)

    def limits_with_margin(self, values, errors, logY=False):
        values[np.isinf(values) | np.isnan(values)] = 0.
        errors[np.isinf(errors) | np.isnan(errors)] = 0.

        if logY:
            positive = values[values > 0.]
            if positive.size:
                self.limits_log(positive.min(), (values + errors).max())
            return

        up = values + errors
        do = values - errors
        diff_step = 0.05 * abs(up.max()-do.min())
        if diff_step > 0.: # avoid UserWarning when the max/min limits are equal (dummy histograms)
            self.limits(y=(do.min() - diff_step, up.max() + 2*diff_step), logY=logY)

    def save(self, name):
        for ext in self.extensions:
            print(" ### INFO: Saving " + name + '.' + ext)
            plt.savefig(name + '.' + ext)
        plt.close()

FLOW_VARS = ('HFEMEt', 'HFHadronEt', 'chargedHadronEt', 'neutralHadronEt', 'photonEt', 'muonEt')

def plot1D(plotter, h, text, props, var=None, name=None):
    """Plot 1D distributions. For the `FLOW_VARS`, the number of underflow and overflow entries is printed (see --show_flow)."""
    nbins, bin_edges, bin_centers, bin_widths = define_bins(h)
    values, errors = histo_values_errors(h)
    errors /= 2

    plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                        fmt='s', color='black', label=props.x, **errorbar_kwargs)
    plotter.ax.stairs(values, bin_edges, color='black', linewidth=2, baseline=None)
    plotter.ax.text(0.03, 0.97, text, transform=plotter.ax.transAxes, fontsize=16,
                    verticalalignment='top', horizontalalignment='left')
    flow_text = None
    if args.show_flow and var in FLOW_VARS:
        underflow, overflow = h.GetBinContent(0), h.GetBinContent(h.GetNbinsX() + 1)
        flow_text = plotter.ax.text(0.03, 0.92, f"Underflow: {round(underflow):d}\nOverflow: {round(overflow):d}",
                        transform=plotter.ax.transAxes, fontsize=14,
                        verticalalignment='top', horizontalalignment='left')

    lims = axis_limits(name or var, props.hist)
    vis = visible(bin_centers, lims)
    plotter.limits_with_margin(values[vis], errors[vis], logY=props.logy)
    plotter.apply_limits(lims)
    plotter.labels(x=props.x, y=props.y)
    if flow_text is not None and 'y' not in lims: # user-defined y limits are respected
        plotter.make_room_above(flow_text, (values + errors)[vis].max(), logY=props.logy)
    return plotter

def plot2D(plotter, h, text, props, name=None):
    """Plot 2D distributions."""
    nbins_x, nbins_y, x_edges, y_edges = define_bins_2D(h)
    values = histo_values_2D(h)

    values = np.where(values == 0., np.nan, values)
    if props.logz:
        values_log = values[~np.isnan(values)] # avoid log(0) errors
        pcm = plotter.ax.pcolormesh(
            x_edges, y_edges, values,
            cmap='viridis',
            shading='auto',
            norm=LogNorm(vmin=values_log.min(), vmax=values_log.max())
        )
    else:
        pcm = plotter.ax.pcolormesh(
            x_edges, y_edges, values,
            cmap='viridis',
            shading='auto'
        )

    plotter.apply_limits(axis_limits(name, props.hist))

    if props.drawdiag:
        diag = np.linspace(*plotter.ax.get_xlim(), 1000)
        plotter.ax.plot(diag, diag, label=r'$y=x$', c='black')
        plotter.labels(x=props.x, y=props.y, legend_title='', legend_loc='best')
    else:
        plotter.labels(x=props.x, y=props.y)

    plotter.fig.colorbar(pcm, ax=plotter.ax, label="# Events")
    return plotter

def plotVars(afile, adir, avars, mode, outdir, metType):
    """
    Plots 1D and 2D distributions.
    The `avars` variables is a dictionary whose values are (xlabel, ylabel, rebin).
    """
    createDir(outdir)
    fontsize = 14
    for var, props in avars.items():
        if props.out is not None:
            createDir(os.path.join(outdir, props.out))

        plotter = Plotter(args.sample_label, period=args.period, fontsize=fontsize, en_com=args.energy)
        root_hist = checkRootFile(afile, f"{adir}/{props.hist or var}", rebinx=props.rebinx, rebiny=props.rebiny)

        if mode == '1D':
            plotter = plot1D(plotter, root_hist, metType, props, var=props.hist or var, name=var)
        elif mode == '2D':
            plotter = plot2D(plotter, root_hist, metType, props, name=var)
        else:
            raise RuntimeError(f'Mode {mode} not supported.')

        plt.tight_layout()
        plotter.save( os.path.join(outdir, props.out, var) )

def plot1DCollectionComparison(afile, adir, avars, outdir, metTypes):
    """
    Plots 1D distributions.
    The `avars` variables is a dictionary whose values are (xlabel, ylabel, rebin).
    """
    outdir = os.path.join(outdir, 'ComparisonCollections_' + '_'.join(metTypes))
    createDir(outdir)

    for outname, props in avars.items():
        var = props.hist or outname
        lims = axis_limits(outname, props.hist)
        plotter = Plotter(args.sample_label, period=args.period, en_com=args.energy)

        amax, amin, pmin = float('-inf'), float('+inf'), float('+inf')
        for metType in metTypes:
            root_hist = checkRootFile(afile, f"{adir}/{metType}/{var}", rebinx=props.rebinx)
            nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
            values, errors = histo_values_errors(root_hist)
            errors /= 2 # symmetrize

            patch = plotter.ax.stairs(values, bin_edges, linewidth=2, baseline=None)
            plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                                fmt='s', label=metType, color=patch.get_edgecolor(),
                                **errorbar_kwargs)
            
            vis = visible(bin_centers, lims)
            amax = max(amax, (values + errors)[vis].max())
            amin = min(amin, (values - errors)[vis].min())
            if (values[vis] > 0.).any():
                pmin = min(pmin, values[vis][values[vis] > 0.].min())

        if props.logy:
            plotter.limits_log(pmin, amax)
        else:
            diff_step = 0.05 * abs(amax-amin)
            plotter.limits(y=(amin - diff_step, amax + 2*diff_step), logY=False)
        plotter.apply_limits(lims)
        plotter.labels(x=props.x, y=props.y)
        plotter.ax.legend(prop={'size': 15})
        plotter.save( os.path.join(outdir, outname) )

def plot1DFilesComparison(adir, avars, outdir, files, files_labels, metType):
    """
    Plots 1D distributions.
    The `avars` variables is a dictionary whose values are (xlabel, ylabel, rebin).
    """
    outdir = os.path.join(outdir, 'ComparisonFiles_' + '_'.join(files_labels), metType)
    createDir(outdir)

    for outname, props in avars.items():
        var = props.hist or outname
        lims = axis_limits(outname, props.hist)
        plotter = Plotter(args.sample_label, period=args.period, en_com=args.energy)

        amax, amin, pmin = float('-inf'), float('+inf'), float('+inf')
        for afile, alabel in zip(files, files_labels):
            afile = ROOT.TFile.Open(afile)
            root_hist = checkRootFile(afile, f"{adir}/{metType}/{var}", rebinx=props.rebinx)
            nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
            values, errors = histo_values_errors(root_hist)
            errors /= 2 # symmetrize

            patch = plotter.ax.stairs(values, bin_edges, linewidth=2, baseline=None)
            plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                                fmt='s', label=alabel, color=patch.get_edgecolor(),
                                **errorbar_kwargs)
            
            vis = visible(bin_centers, lims)
            amax = max(amax, (values + errors)[vis].max())
            amin = min(amin, (values - errors)[vis].min())
            if (values[vis] > 0.).any():
                pmin = min(pmin, values[vis][values[vis] > 0.].min())

        if props.logy:
            plotter.limits_log(pmin, amax)
        else:
            diff_step = 0.05 * abs(amax-amin)
            plotter.limits(y=(amin - diff_step, amax + 2*diff_step), logY=False)
        plotter.apply_limits(lims)
        plotter.labels(x=props.x, y=props.y)
        plotter.ax.legend(prop={'size': 15})
        plotter.save( os.path.join(outdir, outname) )

def overlayLabel(folder):
    """Legend label of a scan folder: `minPt<pt>_maxEta<eta>` or `pdgid<N>`."""
    m = re.fullmatch(r'minPt([\dp]+)_maxEta([\dp]+)', folder)
    if m:
        pt, eta = (v.replace('p', '.') for v in m.groups())
        return rf'$p_T^\text{{jet}} > {pt}$ GeV, $|\eta^\text{{jet}}| < {eta}$'
    m = re.fullmatch(r'pdgid(\d+)', folder)
    if m:
        lep = {'11': r'e', '13': r'\mu', '15': r'\tau'}.get(m.group(1))
        return rf"$W' \to {lep}\nu$" if lep else f'|pdgId| = {m.group(1)}'
    return folder

def subfolders(afile, adir, pattern):
    """Names of the (sorted) sub-directories of `adir` matching the regex `pattern`."""
    d = afile.Get(adir)
    if not d:
        return []
    names = [k.GetName() for k in d.GetListOfKeys()
             if k.IsFolder() and re.fullmatch(pattern, k.GetName())]
    key = lambda n: [float(x.replace('p', '.')) for x in re.findall(r'\d+(?:p\d+)?', n)]
    return sorted(names, key=key)

def plot1DOverlay(afile, adir, folders, avars, outdir, metType, legend_title=None):
    """
    Overlays, for each variable, the normalised distributions stored in the sub-folders `folders` of `adir`.
    Sub-folders without the variable or without entries are skipped.
    """
    createDir(outdir)
    for var, props in avars.items():
        lims = axis_limits(var, props.hist)
        plotter = Plotter(args.sample_label, period=args.period, en_com=args.energy)

        amax, amin, drawn = float('-inf'), float('+inf'), 0
        for folder in folders:
            root_hist = checkRootFile(afile, f"{adir}/{folder}/{var}", rebinx=props.rebinx)
            nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
            values, errors = histo_values_errors(root_hist)
            integral = values.sum()
            if integral <= 0.:
                print(f" ### WARNING: {adir}/{folder}/{var} is empty, skipping.")
                continue
            values, errors = values / integral, errors / integral / 2 # normalise and symmetrize

            patch = plotter.ax.stairs(values, bin_edges, linewidth=2, baseline=None)
            plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                                fmt='s', label=overlayLabel(folder), color=patch.get_edgecolor(),
                                **errorbar_kwargs)
            vis = visible(bin_centers, lims)
            amax = max(amax, (values + errors)[vis].max())
            amin = min(amin, (values - errors)[vis].min())
            drawn += 1

        if drawn == 0:
            plt.close(plotter.fig)
            continue
        diff_step = 0.05 * abs(amax-amin)
        plotter.limits(y=(min(0., amin - diff_step), amax + 2*diff_step), logY=False)
        plotter.apply_limits(lims)
        plotter.labels(x=props.x, y='Fraction of events')
        plotter.ax.legend(title=legend_title, prop={'size': 14}, title_fontsize=15, loc='upper right')
        plotter.save( os.path.join(outdir, var) )

def plotScan(afile, adir, metType, outdir):
    """
    Overlays MEx, MEy and METPhi of the cut scan produced by `addMHTScan` in `<adir>`:
    all cuts in `Overlay_cuts`, and, for each cut, the pdgId split in `Overlay_pdgid/<cut>`.
    """
    avars = {k: v for k, v in varsToPlot(metType)[0].items() if k in ('MEx', 'MEy', 'METPhi')}
    cuts = subfolders(afile, adir, r'minPt[\dp]+_maxEta[\dp]+')
    if not cuts:
        print(f" ### WARNING: no cut-scan folders found in {adir}.")
        return
    title = METType.get(metType, metType)
    plot1DOverlay(afile, adir, cuts, avars, os.path.join(outdir, 'Overlay_cuts'), metType, legend_title=title)
    for cut in cuts:
        pdgs = subfolders(afile, f"{adir}/{cut}", r'pdgid\d+')
        if pdgs:
            plot1DOverlay(afile, f"{adir}/{cut}", pdgs, avars,
                          os.path.join(outdir, 'Overlay_pdgid', cut), metType,
                          legend_title=f"{title}, {overlayLabel(cut)}")

def varsToPlot(metColl):
    """
    Define which variables to plot and the corresponding plot properties.
    """
    vars1D = {
        # MET and MHT common variables
        'MET'                     : HLTMETInputs(x=r'$MET_\text{Reco}$ [GeV]', y=nEvts, rebinx=2, logy=False),
        'GenMETTrue'              : HLTMETInputs(x=r'$MET_\text{Gen}$ [GeV]', y=nEvts, rebinx=2),
        'MEx'                     : HLTMETInputs(x=r'$MET_x$ [GeV]', y=nEvts, rebinx=4),
        'MEy'                     : HLTMETInputs(x=r'$MET_y$ [GeV]', y=nEvts, rebinx=4),
        'METPhi'                  : HLTMETInputs(x=r'$\text{MET}\phi_\text{Reco}$', y=nEvts, rebinx=2),
        'METDeltaPhi_GenMETTrue'  : HLTMETInputs(x=r'$\text{MET}\phi_\text{Reco} - \text{MET}\phi_\text{Gen}$', y=nEvts, rebinx=2),
        'METDiff_GenMETTrue'      : HLTMETInputs(x=r'$\text{MET}_\text{Reco} - \text{MET}_\text{Gen}$ [GeV]', y=nEvts, rebinx=10),
        'METxDiff_GenMETTrue'     : HLTMETInputs(x=r'$\text{MET}_{x,\text{Reco}} - \text{MET}_{x,\text{Gen}}$ [GeV]', y=nEvts, rebinx=10),
        'METyDiff_GenMETTrue'     : HLTMETInputs(x=r'$\text{MET}_{y,\text{Reco}} - \text{MET}_{y,\text{Gen}}$ [GeV]', y=nEvts, rebinx=10),
        'METSignPseudo'           : HLTMETInputs(x='MET / √(∑E_T) "Significance" (event-by-event)', y=nEvts), # \sqrt{} throws a warning
        'METSignReal'             : HLTMETInputs(x='MET Significance (Likelihood)', y=nEvts), # covariance matrix missing
        'MET_Nvtx'                : HLTMETInputs(x='Number of vertices (MET-weighted)', y=nEvts, rebinx=10),
        'Nvertex'                 : HLTMETInputs(x='Number of vertices', y=nEvts, rebinx=10),
        'SumET'                   : HLTMETInputs(x=r'$\sum E_T$ [GeV]', y=nEvts, rebinx=4),
        # log-scale versions: for MHT collections, MET/GenMETTrue are the MHT/genMHT (labels swapped below)
        'MET_logy'                : HLTMETInputs(x=r'$MET_\text{Reco}$ [GeV]', y=nEvts, hist='MET', rebinx=2, logy=True),
        'GenMETTrue_logy'         : HLTMETInputs(x=r'$MET_\text{Gen}$ [GeV]', y=nEvts, hist='GenMETTrue', rebinx=2, logy=True),
        'METDiff_GenMETTrue_logy' : HLTMETInputs(x=r'$\text{MET}_\text{Reco} - \text{MET}_\text{Gen}$ [GeV]', y=nEvts, hist='METDiff_GenMETTrue', rebinx=10, logy=True),
        # MET post-processing
        'METDiffAggr_MET'      : HLTMETInputs(x=r'$\text{MET}_{\text{Gen}}$ [GeV]', y=r'$\langle\text{MET}_\text{Reco} - \text{MET}_\text{Gen}\rangle$ [GeV]'),
        'METDiffAggr_Phi'      : HLTMETInputs(x=r'$\text{MET}\phi_{\text{Gen}}$', y=r'$\langle\text{MET}_\text{Reco} - \text{MET}_\text{Gen}\rangle$ [GeV]'),
        'METDeltaPhiAggr_MET'  : HLTMETInputs(x=r'$\text{MET}_{\text{Gen}}$ [GeV]', y=r'$\langle\text{MET}\phi_\text{Reco} - \text{MET}\phi_\text{Gen}\rangle$'),
        'METDeltaPhiAggr_Phi'  : HLTMETInputs(x=r'$\text{MET}\phi_{\text{Gen}}$', y=r'$\langle\text{MET}\phi_\text{Reco} - \text{MET}\phi_\text{Gen}\rangle$'),
        'METRespAggr_MET'      : HLTMETInputs(x=r'$\text{MET}_{\text{Gen}}$ [GeV]', y=r'$\langle\text{MET}_\text{Reco} / \text{MET}_\text{Gen}\rangle$'),
        'METRespAggr_Phi'      : HLTMETInputs(x=r'$\text{MET}\phi_{\text{Gen}}$', y=r'$\langle\text{MET}_\text{Reco} / \text{MET}_\text{Gen}\rangle$'),
        'METResolAggr_Phi'     : HLTMETInputs(x=r'$\text{MET}\phi_{\text{Gen}}$', y=r'$\sigma(\text{MET}_\text{Reco})$ [GeV]'),
        'METGenResolAggr_Phi'  : HLTMETInputs(x=r'$\text{MET}\phi_{\text{Gen}}$', y=r'$\sigma(\text{MET}_\text{Gen})$ [GeV]'),
        'METResolDiffAggr_Phi' : HLTMETInputs(x=r'$\text{MET}\phi_{\text{Gen}}$', y=r'$\sigma(\text{MET}_\text{Reco}) - \sigma(\text{MET}_\text{Gen})$ [GeV]'),
        'METSignAggr_Phi'      : HLTMETInputs(x=r'$\text{MET}\phi_{\text{Gen}}$', y=r'$\langle\text{MET}_\text{Reco}\rangle / \sigma(\text{MET}_\text{Reco})$'),
        'METGenSignAggr_Phi'   : HLTMETInputs(x=r'$\text{MET}\phi_{\text{Gen}}$', y=r'$\langle\text{MET}_\text{Gen}\rangle / \sigma(\text{MET}_\text{Gen})$'),
        # pileup dependence (no rebinning: the content is a mean or an RMS)
        'METRespAggr_Nvtx'        : HLTMETInputs(x=nvtxLabel, y=r'$\langle\text{MET}_\text{Reco} / \text{MET}_\text{Gen}\rangle$'),
        'METDiffAggr_Nvtx'        : HLTMETInputs(x=nvtxLabel, y=r'$\langle\text{MET}_\text{Reco} - \text{MET}_\text{Gen}\rangle$ [GeV]'),
        'METDiffResolAggr_Nvtx'   : HLTMETInputs(x=nvtxLabel, y=r'$\sigma(\text{MET}_\text{Reco} - \text{MET}_\text{Gen})$ [GeV]'),
        'METxDiffAggr_Nvtx'       : HLTMETInputs(x=nvtxLabel, y=r'$\langle\text{MET}_{x,\text{Reco}} - \text{MET}_{x,\text{Gen}}\rangle$ [GeV]'),
        'METxDiffResolAggr_Nvtx'  : HLTMETInputs(x=nvtxLabel, y=r'$\sigma(\text{MET}_{x,\text{Reco}} - \text{MET}_{x,\text{Gen}})$ [GeV]'),
        'METyDiffAggr_Nvtx'       : HLTMETInputs(x=nvtxLabel, y=r'$\langle\text{MET}_{y,\text{Reco}} - \text{MET}_{y,\text{Gen}}\rangle$ [GeV]'),
        'METyDiffResolAggr_Nvtx'  : HLTMETInputs(x=nvtxLabel, y=r'$\sigma(\text{MET}_{y,\text{Reco}} - \text{MET}_{y,\text{Gen}})$ [GeV]'),
        'METSignDiffAggr_Phi'  : HLTMETInputs(x=r'$\text{MET}\phi_{\text{Gen}}$',
                                              y=r'$\langle\text{MET}_\text{Reco}\rangle / \sigma(\text{MET}_\text{Reco}) - \langle\text{MET}_\text{Gen}\rangle / \sigma(\text{MET}_\text{Gen})$'),
    }

    isMHT = 'MHT' in metColl or metColl.lower().endswith('mht')
    isCalo = metColl in ('hltMet', 'hltCaloMET', 'caloMet')
    isMini = metColl.startswith('slimmed') # miniAOD MET: no Calo gen MET, only the Et fractions of the PF types
    if not isMHT and not isMini:
        vars1D.update({
            # MET-specific
            'GenMETCalo'              : HLTMETInputs(x=r'$MET^\text{Calo}_\text{Gen}$ [GeV]', y=nEvts, rebinx=2),
            'GenMETCalo_logy'         : HLTMETInputs(x=r'$MET^\text{Calo}_\text{Gen}$ [GeV]', y=nEvts, hist='GenMETCalo', rebinx=2, logy=True),
            'METDeltaPhi_GenMETCalo'  : HLTMETInputs(x=r'$\text{MET}\phi_\text{Reco} - \text{MET}\phi^{Calo}_\text{Gen}$', y=nEvts, rebinx=2),
            'METDiff_GenMETCalo'      : HLTMETInputs(x=r'$\text{MET}_\text{Reco} - \text{MET}^\text{Calo}_\text{Gen}$ [GeV]', y=nEvts, rebinx=10),
            'METDeltaPhi_GenMETCalo_logy' : HLTMETInputs(x=r'$\text{MET}\phi_\text{Reco} - \text{MET}\phi^{Calo}_\text{Gen}$', y=nEvts, hist='METDeltaPhi_GenMETCalo', rebinx=2, logy=True),
        })
    if isCalo:
        vars1D.update({
            # CaloMET-specific
            'CaloMaxEtInEmTowers'     : HLTMETInputs(x=r'Max $E_T$ in EM towers [GeV]', y=nEvts),
            'CaloMaxEtInHadTowers'    : HLTMETInputs(x=r'Max $E_T$ in hadronic towers [GeV]', y=nEvts),
            'CaloEtFractionHadronic'  : HLTMETInputs(x=r'Hadronic $E_T$ fraction', y=nEvts),
            'CaloEmEtFraction'        : HLTMETInputs(x=r'EM $E_T$ fraction', y=nEvts),
            'CaloHadEtInHB'           : HLTMETInputs(x=r'Hadronic $E_T$ in HB [GeV]', y=nEvts),
            'CaloHadEtInHE'           : HLTMETInputs(x=r'Hadronic $E_T$ in HE [GeV]', y=nEvts),
            'CaloHadEtInHO'           : HLTMETInputs(x=r'Hadronic $E_T$ in HO [GeV]', y=nEvts),
            'CaloHadEtInHF'           : HLTMETInputs(x=r'Hadronic $E_T$ in HF [GeV]', y=nEvts),
            'CaloEmEtInEB'            : HLTMETInputs(x=r'EM $E_T$ in EB [GeV]', y=nEvts),
            'CaloEmEtInEE'            : HLTMETInputs(x=r'EM $E_T$ in EE [GeV]', y=nEvts),
            'CaloEmEtInHF'            : HLTMETInputs(x=r'EM $E_T$ in HF [GeV]', y=nEvts),
        })
    elif not isMHT:
        pfvars = {
            # PFMET-specific
            'HFEMEt'                  : HLTMETInputs(x=r'HF EM $E_T$  [GeV]', y=nEvts),
            'HFEMEtFraction'          : HLTMETInputs(x=r'HF EM $E_T$ fraction', y=nEvts),
            'HFHadronEt'              : HLTMETInputs(x=r'HF Hadronic $E_T$ [GeV]', y=nEvts, rebinx=2),
            'HFHadronEtFraction'      : HLTMETInputs(x='HF Hadronic $E_T$ fraction', y=nEvts, rebinx=2),
            'chargedHadronEt'         : HLTMETInputs(x=r'Charged Hadron $E_T$ [GeV]', y=nEvts, rebinx=2),
            'chargedHadronEtFraction' : HLTMETInputs(x=r'Charged Hadron $E_T$ fraction', y=nEvts, rebinx=2),
            'neutralHadronEt'         : HLTMETInputs(x=r'Neutral Hadron $E_T$ [GeV]', y=nEvts, rebinx=2),
            'neutralHadronEtFraction' : HLTMETInputs(x=r'Neutral Hadron $E_T$ fraction', y=nEvts, rebinx=2),
            'photonEt'                : HLTMETInputs(x=r'Photon $E_T$ [GeV]', y=nEvts, rebinx=2),
            'photonEtFraction'        : HLTMETInputs(x=r'Photon $E_T$ fraction', y=nEvts, rebinx=2),
            'muonEt'                  : HLTMETInputs(x=r'Muon $E_T$ [GeV]', y=nEvts),
            'muonEtFraction'          : HLTMETInputs(x=r'Muon $E_T$ fraction', y=nEvts),
            'electronEt'              : HLTMETInputs(x=r'Electron $E_T$ [GeV]', y=nEvts),
            'electronEtFraction'      : HLTMETInputs(x=r'Electron $E_T$ fraction', y=nEvts),
        }
        if isMini:
            pfvars = {k: v for k, v in pfvars.items() if k.endswith('Fraction')}
            for unc in ('JetRes', 'JetEn', 'MuonEn', 'ElectronEn', 'TauEn', 'UnclusteredEn', 'PhotonEn'):
                for shift in ('Up', 'Down'):
                    pfvars[f'METUnc_{unc}{shift}'] = HLTMETInputs(
                        x=rf'$\text{{MET}} - \text{{MET}}_\text{{{unc} {shift}}}$ [GeV]', y=nEvts)
        vars1D.update(pfvars)
    if isMHT:
        for k in vars1D:
            vars1D[k].x = vars1D[k].x.replace('MET', 'MHT')
            vars1D[k].y = vars1D[k].y.replace('MET', 'MHT')

    vars2D = {
        'GenMETTruevsMET': HLTMETInputs(x=r'$MET_\text{Gen}$ [GeV]', y=r'$MET_\text{Reco}$ [GeV]', drawdiag=True),
        'GenMETPhivsMET': HLTMETInputs(x=r'$\text{MET}\phi_\text{Gen}$', y=r'$MET_\text{Reco}$ [GeV]'),
        'GenMETTruevsGenMETPhi': HLTMETInputs(x=r'$MET_\text{Gen}$ [GeV]', y=r'$\text{MET}\phi_\text{Gen}$'),
        'METDiffvsGenMETTrue': HLTMETInputs(x=r'$\text{MET}_\text{Reco} - \text{MET}_\text{Gen}$ [GeV]',
                                            y=r'$\text{MET}_\text{Gen}$ [GeV]', rebinx=5, rebiny=5),
        'METDiffvsGenMETPhi': HLTMETInputs(x=r'$\text{MET}_\text{Reco} - \text{MET}_\text{Gen}$ [GeV]',
                                           y=r'$\text{MET}\phi_\text{Gen}$', rebinx=5, rebiny=2),
        'METRatiovsGenMETTrue': HLTMETInputs(x=r'$\text{MET}_\text{Reco} / \text{MET}_\text{Gen}$',
                                             y=r'$\text{MET}_\text{Gen}$ [GeV]', rebinx=5, rebiny=5),
        'METRatiovsGenMETPhi': HLTMETInputs(x=r'$\text{MET}_\text{Reco} / \text{MET}_\text{Gen}$',
                                            y=r'$\text{MET}\phi_\text{Gen}$', rebinx=5, rebiny=2),
        'METDeltaPhivsGenMETTrue': HLTMETInputs(x=r'$\text{MET}\phi_\text{Reco} - \text{MET}\phi_\text{Gen}$',
                                                y=r'$\text{MET}_\text{Gen}$ [GeV]', rebinx=1, rebiny=2),
        'METDeltaPhivsGenMETPhi': HLTMETInputs(x=r'$\text{MET}\phi_\text{Reco} - \text{MET}\phi_\text{Gen}$',
                                               y=r'$\text{MET}\phi_\text{Gen}$'),
        'METRatiovsNvtx': HLTMETInputs(x=r'$\text{MET}_\text{Reco} / \text{MET}_\text{Gen}$', y=nvtxLabel, rebinx=5),
        'METDiffvsNvtx': HLTMETInputs(x=r'$\text{MET}_\text{Reco} - \text{MET}_\text{Gen}$ [GeV]', y=nvtxLabel, rebinx=5),
        'METxDiffvsNvtx': HLTMETInputs(x=r'$\text{MET}_{x,\text{Reco}} - \text{MET}_{x,\text{Gen}}$ [GeV]', y=nvtxLabel, rebinx=5),
        'METyDiffvsNvtx': HLTMETInputs(x=r'$\text{MET}_{y,\text{Reco}} - \text{MET}_{y,\text{Gen}}$ [GeV]', y=nvtxLabel, rebinx=5),
    }
    if isMHT:
        for k in vars2D:
            vars2D[k].x = vars2D[k].x.replace('MET', 'MHT')
            vars2D[k].y = vars2D[k].y.replace('MET', 'MHT')


    return vars1D, vars2D

def varsToPlotMETvsMHT():
    """
    Variables stored in the `METvsMHT/<MET>_<MHT>` folders.
    """
    return {
        'METvsMHT': HLTMETInputs(x=r'$MET_\text{Reco}$ [GeV]', y=r'$MHT_\text{Reco}$ [GeV]', logz=False, drawdiag=True),
        'GenMETTruevsGenMHT': HLTMETInputs(x=r'$MET_\text{Gen}$ [GeV]', y=r'$MHT_\text{Gen}$ [GeV]', drawdiag=True),
    }

if __name__ == '__main__':

    def check_list_length(value):
        if len(value) <= 1:
            raise argparse.ArgumentTypeError("List must have more than one item")
        return value

    class DependencyAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if namespace.met is None:
                parser.error("You must specify the MET collections (--met) before the comparison flags.")
            if option_string == "--compare_files_labels" and not namespace.compare_files:
                parser.error("`--compare_files_labels` requires `--compare_files` to be set")
            if option_string == "--compare_files_labels" and not namespace.met:
                parser.error("`--compare_files_labels` requires `--met` to be set")
            if len(namespace.met) > 1:
                parser.error("When comparing files only one MET collection can be used.")
            if len(values) != len(namespace.compare_files):
                parser.error("`--compare_files_labels` must have the same size as `--compare_files`")
            setattr(namespace, self.dest, values)
        
    full_command = 'sample="Wprime"; python3 Validation/RecoMET/scripts/makeHLTMETValidationPlots.py --file Run/DQM_Wprime_1000.root --odir /eos/user/b/bfontana/www/MET_Valid_TICLv5/${sample} -l ${sample} --compare_collections hltPFMET hltPFPuppiMET hltPFPuppiMETTypeOne'
    parser = argparse.ArgumentParser(description='Make HLT MET validation plots. \nRun all MET paths with\n' + full_command)
    parser.add_argument('-o', '--odir', default="HLTMETValidationPlots", required=False, help='Path to the output directory.')
    parser.add_argument('-l', '--sample_label', default="QCD (200 PU)", required=False, help='Sample label for plotting.')
    parser.add_argument('-s', '--step', default='HLT', choices=sorted(DQM_DIRS),
                        help='Validation step whose MET collections are plotted (default: %(default)s).')
    parser.add_argument('--sample', default=None, choices=sorted(AXIS_LIMITS),
                        help='Sample whose axis limits (AXIS_LIMITS) are applied. If not given, it is inferred from --sample_label '
                             f'(default: {DEFAULT_SAMPLE} if the label names no known sample); an error is raised if both disagree.')
    parser.add_argument('--axis_limits', action=argparse.BooleanOptionalAction, default=True,
                        help='Restrict the plotted range according to AXIS_LIMITS (default: True; '
                             'disable with --no-axis_limits to see the full range).')
    parser.add_argument('--show_flow', action=argparse.BooleanOptionalAction, default=True,
                        help='Print the number of underflow and overflow entries in the HFEMEt, HFHadronEt, '
                             'chargedHadronEt, neutralHadronEt, photonEt and muonEt plots (default: True; '
                             'disable with --no-show_flow).')
    parser.add_argument('--overlay_scan', action='store_true',
                        help='Also overlay the normalised MEx, MEy and METPhi of the cut scan (and its pdgId split) '
                             'found in the sub-folders of each --met collection (see addMHTScan).')
    parser.add_argument('-p', '--period', default="Phase-2", required=False, choices=('Phase-2', 'Run 3'), help='Data-taking period; also sets the center-of-mass energy in the plot label.')

    mutual_excl1 = parser.add_mutually_exclusive_group(required=True)
    mutual_excl1.add_argument('-m', '--met', nargs='+',
                             required=False, help='Name of the met collection(s).')
    mutual_excl1.add_argument('-c', '--compare_collections', nargs='+',
                             type=check_list_length, required=False,
                             help='Name of the met collection(s) to compare, in the same file.', )

    mutual_excl2 = parser.add_mutually_exclusive_group(required=True)
    mutual_excl2.add_argument('-f', '--file', help='Paths to the DQM ROOT file.')
    mutual_excl2.add_argument('-x', '--compare_files', nargs='+', type=check_list_length,
                              help='Compare the same collection in different DQM files.', )
    parser.add_argument('-y', '--compare_files_labels', nargs='+',
                        action=DependencyAction, required=False,
                        help='Labels for the different DQM files being compared. Specify after --met and --compare_files.',)
    
    args = parser.parse_args()

    # Center-of-mass energy is fixed by the period, not a user option
    args.energy = '13.6' if args.period == 'Run 3' else '14'

    try:
        args.sample, why = resolveSample(args.sample, args.sample_label)
    except ValueError as e:
        parser.error(str(e))
    printOptions(args, why)
    if args.overlay_scan and args.step != 'HLT':
        print(" ### WARNING: --overlay_scan is only available for the HLT step, ignoring it.")
        args.overlay_scan = False

    createDir(args.odir)
    if args.met:
        for metType in args.met:
            outdir = createDir(os.path.join(args.odir, metType))
    
    tprofile_rebinning = {'B': (30, 40, 50, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600), #barrel
                          'E': (30, 40, 50, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600), # endcap
                          'F': (30, 40, 50, 80, 120, 240, 600)} # forward

    METType = {
        'hltMet': 'Calo MET',
        'hltCaloMET': 'Calo MET',
        'hltPFMETProducer': 'PF MET',
        'hltPFMETNoMuProducer': 'PF MET NoMu',
        'hltPFMHTTightID': 'PF MHT (Tight ID)',
        'hltMht': 'MHT',
        'hltPFMET': 'PF MET',
        'hltPFPuppiMET': 'PF PUPPI MET',
        'hltPFPuppiMETTypeOne': 'PF Type-1 PUPPI MET',
        'hltPFPuppiMHT': 'PF PUPPI MHT',
        'pfMetPuppi': 'PF PUPPI MET',
        'pfPuppiMHT': 'PF PUPPI MHT',
        'caloMet': 'Calo MET',
        'pfMet': 'PF MET',
        'pfMetT0pc': 'PF Type-0 MET',
        'PfMetT1': 'PF Type-1 MET',
        'PfMetT0pcT1': 'PF Type-0+1 MET',
        'slimmedMETs': 'MiniAOD PF MET',
        'slimmedMETsPuppi': 'MiniAOD PUPPI MET',
    }
    
    colors = hep.style.CMS['axes.prop_cycle'].by_key()['color']
    markers = ('o', 's', 'd')
    errorbar_kwargs = dict(capsize=3, elinewidth=0.8, capthick=2, linewidth=2, linestyle='')

    nEvts = '# Events'
    nvtxLabel = 'Number of vertices'

    if args.compare_collections is not None:
        afile = ROOT.TFile.Open(args.file)
        dqm_dir = DQM_DIRS[args.step]
        checkRootDir(afile, dqm_dir)
        plot1DCollectionComparison(afile, dqm_dir, varsToPlot(args.compare_collections[0])[0],
                                   outdir=args.odir, metTypes=args.compare_collections)

    elif args.compare_files is not None:
        dqm_dir = DQM_DIRS[args.step]
        for afile, alabel in zip(args.compare_files, args.compare_files_labels):
            afile = ROOT.TFile.Open(afile)
            checkRootDir(afile, dqm_dir)
        plot1DFilesComparison(dqm_dir, varsToPlot(args.met[0])[0], outdir=args.odir, metType=args.met[0],
                              files=args.compare_files,
                              files_labels=args.compare_files_labels)

    else:
        afile = ROOT.TFile.Open(args.file)
        dqm_dir = DQM_DIRS[args.step]

        # Plot MET variables
        for metType in args.met:
            dqm_dir_met = os.path.join(dqm_dir, metType)
            checkRootDir(afile, dqm_dir_met)
            plotVars(afile, dqm_dir_met, varsToPlot(metType)[0], outdir=os.path.join(args.odir, metType),
                     mode='1D', metType=METType[metType])
            plotVars(afile, dqm_dir_met, varsToPlot(metType)[1], outdir=os.path.join(args.odir, metType),
                     mode='2D', metType=METType[metType])
            if args.overlay_scan:
                plotScan(afile, dqm_dir_met, metType, outdir=os.path.join(args.odir, metType))

        # Plot MET vs MHT comparisons, one folder per MET-MHT pair
        metvsmht_dir = afile.Get(os.path.join(dqm_dir, 'METvsMHT'))
        if metvsmht_dir:
            for key in metvsmht_dir.GetListOfKeys():
                pair = key.GetName()
                plotVars(afile, os.path.join(dqm_dir, 'METvsMHT', pair), varsToPlotMETvsMHT(), mode='2D',
                         outdir=os.path.join(args.odir, 'METvsMHT', pair), metType=pair.replace('_', ' vs '))
