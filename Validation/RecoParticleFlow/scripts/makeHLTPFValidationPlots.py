#!/usr/bin/env python3

import os
import argparse
import numpy as np
import hist
import re
import utils
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.colors import LogNorm
plt.style.use('tableau-colorblind10')
from dataclasses import dataclass
import array
import ROOT
import mplhep as hep
hep.style.use("CMS")

colorblind_palette = ('#1F77B4', '#AEC7E8', '#FF7F0E', '#FFBB78', '#2CA02C', '#98DF8A', '#D62728',
                      '#FF9896', '#9467BD', '#C5B0D5', '#8C564B', '#C49C94', '#E377C2', '#F7B6D2',
                      '#7F7F7F', '#C7C7C7', '#BCBD22', '#DBDB8D', '#17BECF', '#9EDAE5')
import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def debug(mes):
    print('### INFO: ' + mes)
    
def rate_errorbar_declutter(plotter, eff, err, yaxmin, frac=0.01):
    """
    Filter uncertainties if they lie below the minimum (vertical) axis value.
    Used to plot the filtered points differently, for instance by displaying only the upper uncertainty.
    Uncertainties are trimmed to 1. or 0. if these values are crossed
    (the retrieved Clopper-Pearson uncertainties have been symmetrized in the DQMGenericClient).
    """
    filt = eff-err/2 <= yaxmin
    eff_filt = np.where(filt, np.nan, eff)
    err_filt = np.where(filt, np.nan, err)
    
    up_error = eff_filt + err_filt/2
    transform = blended_transform_factory(plotter.ax.transData, plotter.ax.transAxes)
    up_error = np.where(np.isnan(up_error), frac, up_error) # place at 0.9% above the minimum vertical axis value
    up_error = np.where(up_error != frac, np.nan, up_error)

    filt_limit_one = eff_filt+err_filt/2 > 1.
    filt_limit_zero = eff_filt-err_filt/2 < 0.
    err_hi = np.where(filt_limit_one, 0., err_filt)
    err_lo = np.where(filt_limit_zero, 0., err_filt)
    return eff_filt, (err_lo,err_hi), up_error, transform

def define_bins(h):
    """
    Computes the number of bins, edges, centers and widths of a histogram.
    """
    N = h.GetNbinsX()
    edges = np.array([h.GetBinLowEdge(i+1) for i in range(N)])
    edges = np.append(edges, h.GetBinLowEdge(N+1))
    return N, edges, 0.5*(edges[:-1]+edges[1:]), np.diff(edges)

def define_bins_2D(h):
    Nx = h.GetNbinsX()
    Ny = h.GetNbinsY()
    
    x_edges = np.array([h.GetXaxis().GetBinLowEdge(i+1) for i in range(Nx)])
    x_edges = np.append(x_edges, h.GetXaxis().GetBinUpEdge(Nx))
    
    y_edges = np.array([h.GetYaxis().GetBinLowEdge(j+1) for j in range(Ny)])
    y_edges = np.append(y_edges, h.GetYaxis().GetBinUpEdge(Ny))

    return Nx, Ny, x_edges, y_edges

def histo_values_errors(h):
    N = h.GetNbinsX()
    values = np.array([h.GetBinContent(i+1) for i in range(N)])
    errors = np.array([h.GetBinError(i+1) for i in range(N)])
    return values, errors

def histo_values_2D(h, error=False):
    Nx = h.GetNbinsX()
    Ny = h.GetNbinsY()
    values = np.array([
        [h.GetBinContent(i+1, j+1) for i in range(Nx)]
        for j in range(Ny)
    ])
    return values

@dataclass
class InputArgs:
    xtitle: str    
    ytitle: str
    rebin: tuple = None
    ratio: str = ''
    den: str = ''
    num: str = ''
    legden: str = ''
    legnum: str = ''
    var: str = ''
    name: str = ''
    unit: str = ''
    fit: bool = False
    logy: bool = False
    normalize: bool = False

class Plotter:
    def __init__(self, label, fontsize=18, era='Phase2', grid_color='grey'):
        self._fig, self._ax = plt.subplots(figsize=(10, 10))
        self.fontsize = fontsize
        
        if era == 'Phase2': era='Phase-2'; en='14'
        elif era == 'Run3': era='Run-3'; en='13.6'
        hep.cms.text(f' {era} Simulation Preliminary', ax=self._ax, fontsize=fontsize)
        hep.cms.lumitext(label + f" | {en} TeV", ax=self._ax, fontsize=fontsize)
        if grid_color:
            self._ax.grid(which='major', color=grid_color)
        
        self.extensions = ('png', 'pdf')

    @property
    def fig(self):
        return self._fig

    @property
    def ax(self):
        return self._ax

    def labels(self, x, y, legend_title=None, legend_loc='best'):
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

    def limits_with_margin(self, mdValues, mdErrors, logY=False, logX=False):
        amin, amax = float('inf'), float('-inf')

        if not isinstance(mdValues, (list,tuple)):
            mdValues = [mdValues]
            mdErrors = [mdErrors]
            
        for values, errors in zip(mdValues, mdErrors):
            values[np.isinf(values) | np.isnan(values)] = 0.
            errors[np.isinf(errors) | np.isnan(errors)] = 0.
            
            up = values + errors
            do = values - errors

            diff_step = 0.05 * abs(up.max()-do.min())
            if logY:
                amin = min(amin, (max(values.min(), 1e-5)))
                amax = max(amax, up.max() + 10*diff_step)
            else:
                amin = min(amin, do.min() - diff_step)
                amax = max(amax, up.max() + 2*diff_step)

        self.limits(y=(amin,amax), logY=logY)

    def save(self, name):
        for ext in self.extensions:
            debug('Saving ' + name + '.' + ext)
            plt.savefig(name + '.' + ext)
        plt.close()

def plotProject(h, sample_label, era, props, rebin_edges, outname):
    """
    Project and plot slices of a 2D histogram.
    """
    colors_iter = iter(colorblind_palette)

    valuesList, errorsList = [], []
    fit_params = {} if props.fit else None
    plotter = Plotter(sample_label, grid_color=None, era=era)

    for ibin, (low, high) in enumerate(zip(rebin_edges[:-1],rebin_edges[1:])):
        hproj = h.ProjectionY(h.GetName() + "_proj" + str(ibin),
                              h.GetXaxis().FindBin(low), h.GetXaxis().FindBin(high), "e")
    
        nbins, bin_edges, bin_centers, bin_widths = define_bins(hproj)
        values, errors = histo_values_errors(hproj)
        errors /= 2
        valuesList.append(values)
        errorsList.append(errors)
        
        line = plotter.ax.stairs(values, bin_edges, linewidth=2,
                                 baseline=None, color=next(colors_iter))

        label = f"{low} < {props.var} < {high} {props.unit}"
        plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                            color=line.get_edgecolor(),
                            fmt='s', label=label, **errorbar_kwargs)

        if props.fit and hproj.GetMean() > 0. and hproj.GetRMS() > 0:
            gausTF1 = utils.findBestGaussianCoreFit(hproj, meanForRange=0.9, rmsForRange=0.05, quiet=True)
            nsigmas = 3
            xfunc = np.linspace(gausTF1.GetParameter(1) - nsigmas*abs(gausTF1.GetParameter(2)),
                                gausTF1.GetParameter(1) + nsigmas*abs(gausTF1.GetParameter(2)))
            yfunc = np.array([gausTF1.Eval(xi) for xi in xfunc])
            if not (gausTF1.GetParameter(1) < hproj.GetBinLowEdge(1)
                    or gausTF1.GetParameter(1) > hproj.GetBinLowEdge(hproj.GetNbinsX())):
                plotter.ax.plot(xfunc, yfunc, color=line.get_edgecolor(),
                                linewidth=1., linestyle='-')
            fit_params[(low,high)] = (gausTF1.GetParameter(1), gausTF1.GetParameter(2),
                                      gausTF1.GetParError(1), gausTF1.GetParError(2))
            
    plotter.limits_with_margin(valuesList, errorsList, logY=props.logy)
    
    plotter.labels(x=props.xtitle, y=props.ytitle, legend_title='')    
    plotter.ax.grid(color='grey', axis='x')    
    plt.tight_layout()
    plotter.save(outname)
    return fit_params

def plotOverlay(subdirs, cached_histos, name, match_by_score, sample_label, era, props, outdir):
    """
    Plots 1D distributions, overlaying plots with identical names in different 'subdirs'.
    """
    colors_iter = iter(colorblind_palette)
    
    pattern = r"Score(\d+)p(\d+)"
    matching = "score" if match_by_score else "shared energy fraction"
    replacement = lambda m: f"{matching} = {m.group(1)}.{m.group(2)}"
    
    plotter = Plotter(sample_label, grid_color=None, era=era)
    for sub in subdirs:
        root_hist = cached_histos[f"{sub}/{name}"]

        if props.rebin is not None:
            root_hist = root_hist.Clone(f"{name}" + "_clone")
            root_hist.SetDirectory(0) # detach from file

            if isinstance(props.rebin, (int, float)):
                root_hist = root_hist.Rebin(int(props.rebin), f"{name}" + "_rebin")
            elif hasattr(props.rebin, '__iter__'):
                bin_edges_c = array.array('d', props.rebin)
                root_hist = root_hist.Rebin(len(bin_edges_c) - 1, f"{name}" + "_rebin", bin_edges_c)
            else:
                raise ValueError(f"Unknown type for rebin: {type(props.rebin)}")

        nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
        values, errors = histo_values_errors(root_hist)
        if 'Fake' in name: values = np.array([1-i if i != 0 else np.nan for i in values])
        errors /= 2
    
        line = plotter.ax.stairs(values, bin_edges, linewidth=2,
                                 baseline=None, color=next(colors_iter))

        sublabel = re.sub(pattern, replacement, sub)
        if sublabel == f'{matching} = 1.1': sublabel = f'{matching} = 1.0' # Temporary fix 
        if any(x in name for x in ('Eff', 'Fake', 'Split', 'Merge')):
            eff_filt, (err_filt_lo, err_filt_hi), up_error, transform = rate_errorbar_declutter(plotter, values, errors, 0)
            plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=[err_filt_lo,err_filt_hi],
                            color=line.get_edgecolor(),
                            fmt='s', label=sublabel, **errorbar_kwargs)
            plotter.limits(y=(0,1.1), x=(bin_edges[0], bin_edges[-1]))
        else:
            # if "Response" in name:
            #     plotter.limits(y=(0, 2.0))
            plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                            color=line.get_edgecolor(),
                            fmt='s', label=sublabel, **errorbar_kwargs)
            plotter.limits(x=(bin_edges[0], bin_edges[-1]))

    plotter.labels(x=props.xtitle, y=props.ytitle, legend_title='')
    plotter.ax.grid(color='grey')
    plt.tight_layout()
    plotter.save( os.path.join(outdir, name) )

def plotOverlayRatio(subdirs, cached_histos, num, den, match_by_score, sample_label, era, props, outdir):
    """
    Plots 1D distributions of numerator / denominator.
    """
    colors_iter = iter(colorblind_palette)
    pattern = r"Score(\d+)p(\d+)"
    matching = "score" if match_by_score else "shared energy fraction"
    replacement = lambda m: f"{matching} = {m.group(1)}.{m.group(2)}"
    
    plotter = Plotter(sample_label, grid_color=None, era=era)
    for sub in subdirs:
        hist_num = cached_histos[f"{sub}/{num}"]
        hist_den = cached_histos[f"{sub}/{den}"]

        if props.rebin is not None:
            hist_num = hist_num.Clone(f"{sub}/{num}" + "_clone")
            hist_den = hist_den.Clone(f"{sub}/{den}" + "_clone")
            hist_num.SetDirectory(0) # detach from file
            hist_den.SetDirectory(0) # detach from file

            if isinstance(props.rebin, (int, float)):
                hist_num = hist_num.Rebin(int(props.rebin), f"{sub}/{num}" + "_rebin")
                hist_den = hist_den.Rebin(int(props.rebin), f"{sub}/{den}" + "_rebin")
            elif hasattr(props.rebin, '__iter__'):
                bin_edges_c = array.array('d', props.rebin)
                hist_num = hist_num.Rebin(len(bin_edges_c) - 1, f"{sub}/{num}" + "_rebin", bin_edges_c)
                hist_den = hist_den.Rebin(len(bin_edges_c) - 1, f"{sub}/{den}" + "_rebin", bin_edges_c)
            else:
                raise ValueError(f"Unknown type for rebin: {type(props.rebin)}")
        
        nbins, bin_edges, bin_centers, bin_widths = define_bins(hist_num)
        num_vals, num_errors = histo_values_errors(hist_num)
        den_vals, den_errors = histo_values_errors(hist_den)

        ratio_vals = [s / m if m != 0 else np.nan for s, m in zip(num_vals, den_vals)]
        ratio_errors = [np.sqrt((ds / m)**2 + ((s / m) * (dm / m))**2) / 2 if m != 0 else np.nan
                        for s, ds, m, dm in zip(num_vals, num_errors, den_vals, den_errors)]

        line = plotter.ax.stairs(ratio_vals, bin_edges, linewidth=2,
                                 baseline=None, color=next(colors_iter))

        sublabel = re.sub(pattern, replacement, sub)
        if sublabel == f'{matching} = 1.1': sublabel = f'{matching} = 1.0' # Temporary fix
        plotter.ax.errorbar(bin_centers, ratio_vals, xerr=None, yerr=ratio_errors,
                            color=line.get_edgecolor(),
                            fmt='s', label=sublabel, **errorbar_kwargs)

    # plotter.limits_with_margin(valuesList, errorsList, logY=props['logy'])
    plotter.limits(x=(bin_edges[0], bin_edges[-1]))
    if "Resolution" in props.name:
        plotter.limits(y=(0, 0.7))
    plotter.labels(x=props.xtitle, y=props.ytitle, legend_title='')    
    plotter.ax.grid(color='grey')    
    plt.tight_layout()
    plotter.save( os.path.join(outdir, props.name) )

def plotEffComp1D(cached_histos, title, vars1d, outdir, text, sample_label, era, top_text=False, suffix=''):
    """
    Plots 1D distributions.
    """
    plotter = Plotter(sample_label, grid_color=None, era=era)
    ax2 = plotter.ax.twinx()
    eff_color = '#bd1f01'
    
    valuesList, errorsList = [], []
    colors_iter = iter(('black', 'blue'))

    histo_names = [vars1d.den, vars1d.num, vars1d.ratio]
    leg_names = [vars1d.legden, vars1d.legnum, '']
    rebin = vars1d.rebin
    doNormalize = vars1d.normalize
    logy = vars1d.logy
    xlabel = vars1d.xtitle
    ylabel = vars1d.ytitle

    for name, leglabel in zip(histo_names, leg_names):

        root_hist = cached_histos[name]

        if rebin is not None:
            root_hist = root_hist.Clone(f"{root_hist.GetName()}_clone")
            root_hist.SetDirectory(0) # detach from file

            if isinstance(rebin, (int, float)):
                root_hist = root_hist.Rebin(int(rebin), f"{root_hist.GetName()}_rebin")
            elif hasattr(rebin, '__iter__'):
                bin_edges_c = array.array('d', rebin)
                root_hist = root_hist.Rebin(len(bin_edges_c) - 1, f"{root_hist.GetName()}_rebin", bin_edges_c)
            else:
                raise ValueError(f"Unknown type for rebin: {type(rebin)}")

        nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
        values, errors = histo_values_errors(root_hist)
        errors /= 2

        # normalization
        if doNormalize:
            ylabel = '[a.u.]'
        if 'Eff' not in name and doNormalize and sum(values) > 0:
            errors /= sum(values)
            values /= sum(values)
        if 'Fake' in name:
            values = np.array([1-i if i != 0 else np.nan for i in values])
            
        if any(x in name for x in ('Eff', 'Fake', 'Split', 'Merge')):
            if 'Eff' in name:
                axis_name = 'Efficiency'
            elif 'Fake' in name:
                axis_name = 'Fake Rate'
            elif 'Split' in name:
                axis_name = 'Split Rate'
            elif 'Merge' in name:
                axis_name = 'Merge Rate'
            ax2.set_ylabel(axis_name, color=eff_color)
            ax2.set_ylim(-0.05, 1.15)

            eff_filt, (err_filt_lo, err_filt_hi), up_error, transform = rate_errorbar_declutter(plotter, values, errors, 0)
            ax2.stairs(eff_filt, bin_edges, linewidth=2, baseline=None, color=eff_color)
            ax2.errorbar(bin_centers, eff_filt, xerr=None, yerr=[err_filt_lo,err_filt_hi],
                         color=eff_color, fmt='s', label=leglabel, **errorbar_kwargs)

        else:
            line = plotter.ax.stairs(values, bin_edges, linewidth=2,
                                     baseline=None, color=next(colors_iter))
            plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                                color=line.get_edgecolor(),
                                fmt='s', label=leglabel, **errorbar_kwargs)

        valuesList.append(values)
        errorsList.append(errors)

    plotter.limits_with_margin(valuesList, errorsList, logY=logy)
    #plotter.limits(x=(-9,109), y=(10E-5, 10E4), logY=logy)
    plotter.labels(x=xlabel, y=ylabel, legend_title='')

    plotter.ax.text(0.03, 0.97, text, transform=plotter.ax.transAxes, fontsize=fontsize,
                    verticalalignment='top', horizontalalignment='left')

    if top_text:
        plotter.ax.text(0.97, 0.97, root_hist.GetTitle().replace('ET', r'$E_T$'),
                        transform=plotter.ax.transAxes, fontsize=fontsize,
                        verticalalignment='top', horizontalalignment='right')
    
    ax2.grid(color=eff_color, axis='y')
    ax2.tick_params(axis='y', labelcolor=eff_color)
    plotter.ax.grid(color=eff_color, axis='x')
    
    plt.tight_layout()
    plotter.save( os.path.join(outdir, title) )

def plot2D(h, sample_label, era, props, outname):
    """
    Plot with mplhep's hist2d (preserves ROOT bin edges, color bar included)
    empty bins will be invisible (background color).
    """
    plotter = Plotter(sample_label, fontsize=15, era=era)

    nbins_x, nbins_y, x_edges, y_edges = define_bins_2D(h)
    values = histo_values_2D(h)

    values = np.where(values == 0., np.nan, values)
    if 'logz' in props and props['logz']:
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

    plotter.labels(x=props['xtitle'], y=props['ytitle'])
    plotter.fig.colorbar(pcm, ax=plotter.ax, label=props['var'])
    plt.tight_layout()
    plotter.save(outname)
        
def plot1D(h, sample_label, era, props, outname):

    plotter = Plotter(sample_label, fontsize=15, era=era)
    nbins, bin_edges, bin_centers, bin_widths = define_bins(h)
    values, errors = histo_values_errors(h)
    errors /= 2

    plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                        fmt='s', color='black', label=props['var'], **errorbar_kwargs)
    plotter.ax.stairs(values, bin_edges, color='black', linewidth=2, baseline=None)

    plotter.limits_with_margin(values, errors, logY=props['logy'])
    plotter.labels(x=props['xtitle'], y=props['ytitle'])

    plt.tight_layout()
    plotter.save( os.path.join(outname) )


if __name__ == '__main__':

    def check_list_length(value):
        if len(value) <= 1:
            raise argparse.ArgumentTypeError("List must have more than one item")
        return value

    class DependencyAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if option_string == "--compare_files_labels" and not namespace.compare_files:
                parser.error("`--compare_files_labels` requires `--compare_files` to be set")
            if option_string == "--compare_files_labels" and not namespace.met:
                parser.error("`--compare_files_labels` requires `--met` to be set")
            if len(namespace.met) > 1:
                parser.error("When comparing files only one MET collection can be used.")
            if len(values) != len(namespace.compare_files):
                parser.error("`--compare_files_labels` must have the same size as `--compare_files`")
            setattr(namespace, self.dest, values)
        
    full_command = 'python3 Validation/RecoParticleFlow/scripts/makeHLTPFValidationPlots.py --odir <your_dir> -l TTbar -f <your_ROOT_file>'
    parser = argparse.ArgumentParser(description='Make HLT PF validation plots. \nExample command:\n' + full_command)
    parser.add_argument('-o', '--odir', default="HLTPFValidationPlots", help='Path to the output directory.')
    parser.add_argument('-l', '--sample_label', default="", help='Sample label for plotting.')
    parser.add_argument('-e', '--era', default="Phase2", help="Chose between ['Phase2', 'Run3'].")
    parser.add_argument('--EnFracCut', default=0.01, help='Cut on the sim cluster energy fraction.')
    parser.add_argument('--PtCut', default=0.1, help='Cut on the sim cluster energy fraction.')
    parser.add_argument('--match_by_score', default=1, type=int, help='Use association based on score (if false, use shared energy fraction).')
    parser.add_argument('--ticl', default=False, action='store_true', help='Use TiclBarrel folder.')

    mutual_excl2 = parser.add_mutually_exclusive_group(required=True)
    mutual_excl2.add_argument('-f', '--file', help='Paths to the DQM ROOT file.')
    mutual_excl2.add_argument('-x', '--compare_files', nargs='+', type=check_list_length,
                              help='Compare the same collection in different DQM files.', )
    parser.add_argument('-y', '--compare_files_labels', nargs='+',
                        action=DependencyAction, help='Compare the same collection in different DQM files.',)
    
    args = parser.parse_args()

    utils.createDir(args.odir)
    parentDir = os.path.dirname(args.odir)
    if args.odir[-1] == '/':
        parentDir = os.path.dirname(parentDir)
    utils.createIndexPHP(src=parentDir, dest=args.odir)
    
    fontsize = 16    
    colors = hep.style.CMS['axes.prop_cycle'].by_key()['color']
    markers = ('o', 's', 'd')
    errorbar_kwargs = dict(capsize=3, elinewidth=0.8, capthick=2, linewidth=2, linestyle='')

    nSimClustersLabel = '# SimClusters'
    nPFClustersLabel = '# PFClusters'

    titles = {'responsePt': r"$p_{T}^{Reco}/p_{T}^{Sim}$", 
              'response': r"$E^{Reco}/E^{Sim}$", 
              'av_responsePt': r"$<p_{T}^{Reco}/p_{T}^{Sim}>$", 
              'av_response': r"$<E^{Reco}/E^{Sim}>$", 
              'resolutionPt': r"$\sigma(p_{T}^{Reco}/p_{T}^{Sim}) / <p_{T}^{Reco}/p_{T}^{Sim}>$", 
              'resolution': r"$\sigma(E^{Reco}/E^{Sim}) / <E^{Reco}/E^{Sim}>$", 
              'eff': 'Efficiency',
              'fake': 'Fake Rate',
              'split': 'Split Rate',
              'merge': 'Merge Rate'}

    if args.match_by_score == 1:
        matching = 'MatchByScore'
    else:
        matching = 'MatchByShEnF'
        print("### INFO: Using association by shared energy fraction.")
    if args.ticl:   sub_folder = 'TiclBarrel'
    else:           sub_folder = 'ParticleFlow'
    dqm_dir = f"DQMData/Run 1/HLT/Run summary/{sub_folder}/{matching}/PFClusterValidation"
    afile = ROOT.TFile.Open(args.file)
    utils.checkRootDir(afile, dqm_dir)

    debug('Start caching PFCluster histograms...')
    subdirs = []
    cached_histos = {}
    directory = afile.GetDirectory(dqm_dir)
    for key in directory.GetListOfKeys():
        obj = key.ReadObj()
        if isinstance(obj, ROOT.TDirectory):
            subdirs.append(obj.GetName())
            subdir = afile.GetDirectory(f"{dqm_dir}/{obj.GetName()}")
            for subkey in subdir.GetListOfKeys():
                name = subkey.GetName()
                cached_histos[f"{obj.GetName()}/{name}"] = subkey.ReadObj()
        else:
            name = key.GetName()
            cached_histos[name] = key.ReadObj()
    debug(' ...done.')

    # create and setup folders
    for subdir in subdirs:
        utils.checkRootDir(afile, f"{dqm_dir}/{subdir}")
        utils.createDir(f'{args.odir}/{subdir}')
        utils.createIndexPHP(src=args.odir, dest=f'{args.odir}/{subdir}')

    for subdir in subdirs:       
        varsDict = {
            # Cluster efficiency
            f'{subdir}/Eff_vs_En': InputArgs(
                ratio=f'{subdir}/Eff_vs_En', 
                den=f'SimClustersEn', legden='SimClusters',
                num=f'{subdir}/SimClustersMatchedRecoClustersEn', legnum='Matched SimClusters',
                xtitle='SimCluster Energy [GeV]', ytitle=nSimClustersLabel, rebin=4, logy=False, normalize=False
            ),
            f'{subdir}/Eff_vs_EnFrac': InputArgs(
                ratio=f'{subdir}/Eff_vs_EnFrac', 
                den=f'SimClustersEnFrac', legden='SimClusters',
                num=f'{subdir}/SimClustersMatchedRecoClustersEnFrac', legnum='Matched SimClusters', 
                xtitle='Energy Fraction', ytitle=nSimClustersLabel, rebin=4, logy=False, normalize=False
            ),
            f'{subdir}/Eff_vs_EnSimTrack': InputArgs(
                ratio=f'{subdir}/Eff_vs_EnSimTrack', 
                den=f'SimClustersEnSimTrack', legden='SimClusters',
                num=f'{subdir}/SimClustersMatchedRecoClustersEnSimTrack', legnum='Matched SimClusters', 
                xtitle='SimTrack Energy [GeV]', ytitle=nSimClustersLabel, rebin=4, logy=False, normalize=False
            ),
            f'{subdir}/Eff_vs_Pt': InputArgs(
                ratio=f'{subdir}/Eff_vs_Pt', 
                den=f'SimClustersPt', legden='SimClusters',
                num=f'{subdir}/SimClustersMatchedRecoClustersPt', legnum='Matched SimClusters', 
                xtitle=r'$p_{T}$ [GeV]', ytitle=nSimClustersLabel, rebin=4, logy=False, normalize=False
            ),
            f'{subdir}/Eff_vs_Eta': InputArgs(
                ratio=f'{subdir}/Eff_vs_Eta', 
                den=f'SimClustersEta', legden='SimClusters',
                num=f'{subdir}/SimClustersMatchedRecoClustersEta', legnum='Matched SimClusters', 
                xtitle=r'$\eta$', ytitle=nSimClustersLabel, rebin=None, logy=False, normalize=False
            ),
            f'{subdir}/Eff_vs_Phi': InputArgs(
                ratio=f'{subdir}/Eff_vs_Phi', 
                den=f'SimClustersPhi', legden='SimClusters',
                num=f'{subdir}/SimClustersMatchedRecoClustersPhi', legnum='Matched SimClusters', 
                xtitle=r'$\phi$', ytitle=nSimClustersLabel, rebin=None, logy=False, normalize=False
            ),
            f'{subdir}/Eff_vs_Mult': InputArgs(
                ratio=f'{subdir}/Eff_vs_Mult', 
                den=f'SimClustersMult', legden='SimClusters',
                num=f'{subdir}/SimClustersMatchedRecoClustersMult', legnum='Matched SimClusters', 
                xtitle='Multiplicity', ytitle=nSimClustersLabel, rebin=4, logy=False, normalize=False
            ),
            # Cluster split rate
            f'{subdir}/Split_vs_En': InputArgs(
                ratio=f'{subdir}/Split_vs_En', 
                den=f'SimClustersEn', legden='SimClusters',
                num=f'{subdir}/SimClustersMultiMatchedRecoClustersEn', legnum='Multi Matched SimClusters', 
                xtitle='SimCluster Energy [GeV]', ytitle=nSimClustersLabel, rebin=4, logy=False, normalize=False
            ),
            f'{subdir}/Split_vs_EnFrac': InputArgs(
                ratio=f'{subdir}/Split_vs_EnFrac', 
                den=f'SimClustersEnFrac', legden='SimClusters',
                num=f'{subdir}/SimClustersMultiMatchedRecoClustersEnFrac', legnum='Multi Matched SimClusters', 
                xtitle='Energy Fraction', ytitle=nSimClustersLabel, rebin=4, logy=False, normalize=False
            ),
            f'{subdir}/Split_vs_EnSimTrack': InputArgs(
                ratio=f'{subdir}/Split_vs_EnSimTrack', 
                den=f'SimClustersEnSimTrack', legden='SimClusters',
                num=f'{subdir}/SimClustersMultiMatchedRecoClustersEnSimTrack', legnum='Multi Matched SimClusters', 
                xtitle='SimTrack Energy [GeV]', ytitle=nSimClustersLabel, rebin=4, logy=False, normalize=False
            ),
            f'{subdir}/Split_vs_Pt': InputArgs(
                ratio=f'{subdir}/Split_vs_Pt', 
                den=f'SimClustersPt', legden='SimClusters',
                num=f'{subdir}/SimClustersMultiMatchedRecoClustersPt', legnum='Multi Matched SimClusters', 
                xtitle=r'$p_{T}$ [GeV]', ytitle=nSimClustersLabel, rebin=4, logy=False, normalize=False
            ),
            f'{subdir}/Split_vs_Eta': InputArgs(
                ratio=f'{subdir}/Split_vs_Eta', 
                den=f'SimClustersEta', legden='SimClusters',
                num=f'{subdir}/SimClustersMultiMatchedRecoClustersEta', legnum='Multi Matched SimClusters', 
                xtitle=r'$\eta$', ytitle=nSimClustersLabel, rebin=None, logy=False, normalize=False
            ),
            f'{subdir}/Split_vs_Phi': InputArgs(
                ratio=f'{subdir}/Split_vs_Phi', 
                den=f'SimClustersPhi', legden='SimClusters',
                num=f'{subdir}/SimClustersMultiMatchedRecoClustersPhi', legnum='Multi Matched SimClusters', 
                xtitle=r'$\phi$', ytitle=nSimClustersLabel, rebin=None, logy=False, normalize=False
            ),
            f'{subdir}/Split_vs_Mult': InputArgs(
                ratio=f'{subdir}/Split_vs_Mult', 
                den=f'SimClustersMult', legden='SimClusters',
                num=f'{subdir}/SimClustersMultiMatchedRecoClustersMult', legnum='Multi Matched SimClusters', 
                xtitle='Multiplicity', ytitle=nSimClustersLabel, rebin=4, logy=False, normalize=False
            ),
        }

        # Compare pairs of variables
        for title, props in varsDict.items():
            plotEffComp1D(cached_histos, title, vars1d=props, outdir=args.odir, text='',
                          sample_label=args.sample_label, era=args.era, suffix=f'')

        varsDict = {
            # Cluster fake rate
            f'{subdir}/Fake_vs_En': InputArgs(
                ratio=f'{subdir}/Fake_vs_En', 
                den=f'RecoClustersEn', legden='RecoClusters',
                num=f'{subdir}/RecoClustersMatchedSimClustersEn', legnum='Matched RecoClusters', 
                xtitle='Energy [GeV]', ytitle=nPFClustersLabel, rebin=4, logy=True,
            ),
            f'{subdir}/Fake_vs_Pt': InputArgs(
                ratio=f'{subdir}/Fake_vs_Pt', 
                den=f'RecoClustersPt', legden='RecoClusters',
                num=f'{subdir}/RecoClustersMatchedSimClustersPt', legnum='Matched RecoClusters', 
                xtitle=r'$p_{T}$ [GeV]', ytitle=nPFClustersLabel, rebin=4,
            ),
            f'{subdir}/Fake_vs_Eta': InputArgs(
                ratio=f'{subdir}/Fake_vs_Eta', 
                den=f'RecoClustersEta', legden='RecoClusters',
                num=f'{subdir}/RecoClustersMatchedSimClustersEta', legnum='Matched RecoClusters', 
                xtitle=r'$\eta$', ytitle=nPFClustersLabel,
            ),
            f'{subdir}/Fake_vs_Phi': InputArgs(
                ratio=f'{subdir}/Fake_vs_Phi', 
                den=f'RecoClustersPhi', legden='RecoClusters',
                num=f'{subdir}/RecoClustersMatchedSimClustersPhi', legnum='Matched RecoClusters', 
                xtitle=r'$\phi$', ytitle=nPFClustersLabel,
            ),
            f'{subdir}/Fake_vs_Mult': InputArgs(
                ratio=f'{subdir}/Fake_vs_Mult', 
                den=f'RecoClustersMult', legden='RecoClusters',
                num=f'{subdir}/RecoClustersMatchedSimClustersMult', legnum='Matched RecoClusters', 
                xtitle='Multiplicity', ytitle=nPFClustersLabel, rebin=4,
            ),
            # Cluster merge rate (WIP)
            f'{subdir}/Merge_vs_En': InputArgs(
                ratio=f'{subdir}/Merge_vs_En', 
                den=f'RecoClustersEn', legden='RecoClusters',
                num=f'{subdir}/RecoClustersMultiMatchedSimClustersEn', legnum='Multi Matched RecoClusters', 
                xtitle='Energy [GeV]', ytitle=nPFClustersLabel, rebin=4),
            f'{subdir}/Merge_vs_Pt': InputArgs(
                ratio=f'{subdir}/Merge_vs_Pt', 
                den=f'RecoClustersPt', legden='RecoClusters',
                num=f'{subdir}/RecoClustersMultiMatchedSimClustersPt', legnum='Multi Matched RecoClusters', 
                xtitle=r'$p_{T}$ [GeV]', ytitle=nPFClustersLabel, rebin=4
            ),
            f'{subdir}/Merge_vs_Eta': InputArgs(
                ratio=f'{subdir}/Merge_vs_Eta', 
                den=f'RecoClustersEta', legden='RecoClusters',
                num=f'{subdir}/RecoClustersMultiMatchedSimClustersEta', legnum='Multi Matched RecoClusters', 
                xtitle=r'$\eta$', ytitle=nPFClustersLabel
            ),
            f'{subdir}/Merge_vs_Phi': InputArgs(
                ratio=f'{subdir}/Merge_vs_Phi', 
                den=f'RecoClustersPhi', legden='RecoClusters',
                num=f'{subdir}/RecoClustersMultiMatchedSimClustersPhi', legnum='Multi Matched RecoClusters', 
                xtitle=r'$\phi$', ytitle=nPFClustersLabel
            ),
            f'{subdir}/Merge_vs_Mult': InputArgs(
                ratio=f'{subdir}/Merge_vs_Mult', 
                den=f'RecoClustersMult', legden='RecoClusters',
                num=f'{subdir}/RecoClustersMultiMatchedSimClustersMult', legnum='Multi Matched RecoClusters', 
                xtitle='Multiplicity', ytitle=nPFClustersLabel, rebin=4
            ),
        }

        # Compare pairs of variables
        for title, props in varsDict.items():
            plotEffComp1D(cached_histos, title, vars1d=props, outdir=args.odir, text='',
                          sample_label=args.sample_label, era=args.era, suffix=f'')

    varsOverlay = {
        "ResponseE_En_Mean"             : InputArgs(ytitle=titles['response'], rebin=(0., 5., 10., 20., 40., 60., 100.),
                                                    xtitle='SimCluster Energy [GeV]'),
        "ResponseE_EnFrac_Mean"         : InputArgs(ytitle=titles['response'], rebin=(0., 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.),
                                                    xtitle='Energy Fraction'),
        "ResponseE_EnSimTrack_Mean"         : InputArgs(ytitle=titles['response'], rebin=(0., 5., 10., 20., 40., 60., 100.),
                                                    xtitle='SimTrack Energy [GeV]'),
        "ResponseE_Pt_Mean"             : InputArgs(ytitle=titles['response'], rebin=(0., 5., 10., 20., 40., 60., 100.), xtitle=r'$p_{T} [GeV]$'),
        "ResponseE_Eta_Mean"            : InputArgs(ytitle=titles['response'], xtitle=r'$\eta$'),
        "ResponseE_Phi_Mean"            : InputArgs(ytitle=titles['response'], xtitle=r'$\phi$'),
        "ResponseE_Mult_Mean"           : InputArgs(ytitle=titles['response'], rebin=4, xtitle='Multiplicity'),
        "Eff_vs_En"                     : InputArgs(ytitle=titles['eff'], rebin=4, xtitle='SimCluster Energy [GeV]'),
        "Eff_vs_EnFrac"                 : InputArgs(ytitle=titles['eff'], rebin=4, xtitle='Energy Fraction'),
        "Eff_vs_EnSimTrack"             : InputArgs(ytitle=titles['eff'], rebin=6, xtitle='SimTrack Energy'),
        "Eff_vs_Pt"                     : InputArgs(ytitle=titles['eff'], rebin=4, xtitle='$p_{T} [GeV]$'),
        "Eff_vs_Eta"                    : InputArgs(ytitle=titles['eff'], xtitle=r'$\eta$'),
        "Eff_vs_Phi"                    : InputArgs(ytitle=titles['eff'], xtitle=r'$\phi$'),
        "Eff_vs_Mult"                   : InputArgs(ytitle=titles['eff'], rebin=4, xtitle='Multiplicity'),
        "Split_vs_En"                   : InputArgs(ytitle=titles['split'], rebin=4, xtitle='SimCluster Energy [GeV]'),
        "Split_vs_EnFrac"               : InputArgs(ytitle=titles['split'], rebin=6, xtitle='Energy Fraction'),
        "Split_vs_EnSimTrack"           : InputArgs(ytitle=titles['split'], rebin=4, xtitle='SimTrack Energy [GeV]'),
        "Split_vs_Pt"                   : InputArgs(ytitle=titles['split'], rebin=4, xtitle='$p_{T} [GeV]$'),
        "Split_vs_Eta"                  : InputArgs(ytitle=titles['split'], xtitle=r'$\eta$'),
        "Split_vs_Phi"                  : InputArgs(ytitle=titles['split'], xtitle=r'$\phi$'),
        "Split_vs_Mult"                 : InputArgs(ytitle=titles['split'], rebin=4, xtitle='Multiplicity'),
        "Fake_vs_En"                    : InputArgs(ytitle=titles['fake'], rebin=4, xtitle='Energy [GeV]'),
        "Fake_vs_Pt"                    : InputArgs(ytitle=titles['fake'], rebin=4, xtitle='$p_{T} [GeV]$'),
        "Fake_vs_Eta"                   : InputArgs(ytitle=titles['fake'], xtitle=r'$\eta$'),
        "Fake_vs_Phi"                   : InputArgs(ytitle=titles['fake'], xtitle=r'$\phi$'),
        "Fake_vs_Mult"                  : InputArgs(ytitle=titles['fake'], rebin=4, xtitle='Multiplicity'),
        "Merge_vs_En"                   : InputArgs(ytitle=titles['merge'], rebin=4, xtitle='Energy [GeV]'),
        "Merge_vs_Pt"                   : InputArgs(ytitle=titles['merge'], rebin=4, xtitle='$p_{T} [GeV]$'),
        "Merge_vs_Eta"                  : InputArgs(ytitle=titles['merge'], xtitle=r'$\eta$'),
        "Merge_vs_Phi"                  : InputArgs(ytitle=titles['merge'], xtitle=r'$\phi$'),
        "Merge_vs_Mult"                 : InputArgs(ytitle=titles['merge'], rebin=4, xtitle='Multiplicity'),
        }
    for name, props in varsOverlay.items():
        plotOverlay(subdirs, cached_histos, name, args.match_by_score, args.sample_label, args.era, props, outdir=args.odir)

    varsResponse = {
        ("ResponseE_En_Sigma", "ResponseE_En_Mean"):
        InputArgs(
            name='ResolutionEn', ytitle=titles['resolution'], xtitle=r'SimCluster Energy [GeV]',
            rebin=(0., 5., 10., 20., 40., 60., 100.)
        ),
        ("ResponseE_EnFrac_Sigma", "ResponseE_EnFrac_Mean"):
        InputArgs(
            name='ResolutionEnFrac', ytitle=titles['resolution'], xtitle=r'Energy Fraction',
            rebin=(0., 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.)
        ),
        ("ResponseE_EnSimTrack_Sigma", "ResponseE_EnSimTrack_Mean"):
        InputArgs(
            name='ResolutionEnSimTrack', ytitle=titles['resolution'], xtitle=r'SimTrack Energy [GeV]',
            rebin=(0., 5., 10., 20., 40., 60., 100.)
        ),
        ("ResponseE_Pt_Sigma", "ResponseE_Pt_Mean"):
        InputArgs(
            name='ResolutionPt', ytitle=titles['resolution'], xtitle=r'$p_{T} [GeV]$',
            rebin=(0., 5., 10., 20., 40., 60., 100.)
        ),
        ("ResponseE_Eta_Sigma", "ResponseE_Eta_Mean"):
        InputArgs(
            name='ResolutionEta', ytitle=titles['resolution'], xtitle=r'$\eta$'
        ),
        ("ResponseE_Phi_Sigma", "ResponseE_Phi_Mean"):
        InputArgs(
            name='ResolutionPhi', ytitle=titles['resolution'], xtitle=r'$\phi$'
        ),
        ("ResponseE_Mult_Sigma", "ResponseE_Mult_Mean"):
        InputArgs(
            name='ResolutionMult', ytitle=titles['resolution'], xtitle='Multiplicity', rebin=4
        ),
    }
    for (num, den), props in varsResponse.items():
        plotOverlayRatio(subdirs, cached_histos, num, den, args.match_by_score, args.sample_label, args.era, props, outdir=args.odir)

    vars2DProjection = {
        **{f'{subdir}/ResponseE_En':
           InputArgs(
               fit=True, xtitle=titles['response'], ytitle='# Clusters', var=r'E', unit='[GeV]',
               rebin=(0., 20., 40., 60, 80., 100.)
           ) for subdir in subdirs},
        **{f'{subdir}/ResponseE_EnFrac':
           InputArgs(
               xtitle=titles['response'], ytitle='# Clusters', var=r'Energy Fraction',
               rebin=(0., 0.1, 0.5, 0.9, 1.)
           ) for subdir in subdirs},
        **{f'{subdir}/ResponseE_EnSimTrack':
           InputArgs(
               xtitle=titles['response'], ytitle='# Clusters', var=r'$E_{SimTrack}$', unit='[GeV]',
               rebin=(0., 20., 40., 60, 80., 100.)
           ) for subdir in subdirs},
        **{f'{subdir}/ResponseE_Pt':
           InputArgs(
               xtitle=titles['response'], ytitle='# Clusters', var=r'$p_{T}$', unit='[GeV]',
               rebin=(0., 20., 40., 60, 80., 100.)
           ) for subdir in subdirs},
        **{f'{subdir}/ResponseE_Eta':
           InputArgs(
               xtitle=titles['response'], ytitle='# Clusters', var=r'$\eta$',
               # rebin=(-1.5, -1.3, -1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 1., 1.3, 1.5)
               rebin=(-1.5, -0.75, 0., 0.75, 1.5)
           ) for subdir in subdirs},
        **{f'{subdir}/ResponseE_Phi':
           InputArgs(
               xtitle=titles['response'], ytitle='# Clusters', var=r'$\phi$',
               rebin=(-3.15, -1.5, 0., 1.5, 3.15)
           ) for subdir in subdirs},
        **{f'{subdir}/ResponseE_Mult':
           InputArgs(
               xtitle=titles['response'], ytitle='# Clusters', var='Multiplicity',
               rebin=(0, 50, 100, 150, 200)
           ) for subdir in subdirs},
        'SimClustersEnFrac_Mult':
        InputArgs(
            xtitle='Multiplicity', ytitle='# Clusters', var='Energy Fraction',
            logy=True, rebin=(0., 0.005, 0.01, 0.02, 0.03, 1.)
        ),
    }

    for name, props in vars2DProjection.items():
        root_hist = cached_histos[f"{name}"]
        fitpars = plotProject(root_hist, args.sample_label, args.era, props, rebin_edges=props.rebin,
                              outname=os.path.join(args.odir, name + '_Projected'))

        if props.fit:
            n_bins = len(fitpars)
            xmin = min(low for low, _ in fitpars.keys())
            xmax = max(high for _, high in fitpars.keys())

            fitstr = 'FromFit{}_'+os.path.basename(name)
            fitdir = os.path.dirname(name) + '/' + fitstr
            cached_histos[fitdir.format('Mean')] = ROOT.TH1F(fitdir.format('Mean').replace('/','_'), fitstr.format('Mean'), n_bins, xmin, xmax)
            cached_histos[fitdir.format('Width')] = ROOT.TH1F(fitdir.format('Width').replace('/','_'), fitstr.format('Width'), n_bins, xmin, xmax)

            for i, ((low, high), (mean, width, mean_err, width_err)) in enumerate(fitpars.items(), 1):
                cached_histos[fitdir.format('Mean')].SetBinContent(i, mean)
                cached_histos[fitdir.format('Mean')].SetBinError(i, mean_err)
                cached_histos[fitdir.format('Width')].SetBinContent(i, width)
                cached_histos[fitdir.format('Width')].SetBinError(i, width_err)
                
    for name, props in vars2DProjection.items():
        if props.fit:
            props.xtitle = 'Energy [GeV]'
            props.ytitle = 'Response'
            plotOverlay(subdirs, cached_histos, fitstr.format('Mean'), args.match_by_score, args.sample_label, args.era, props, outdir=args.odir)
            props.ytitle = 'Resolution'
            plotOverlay(subdirs, cached_histos, fitstr.format('Width'), args.match_by_score, args.sample_label, args.era, props, outdir=args.odir)

    vars2D = {
        'SimClustersEnFrac_Mult': dict(ytitle='Multiplicity', var='# Sim Clusters', xtitle='Energy Fraction'),
        'SimClustersEn_Mult': dict(ytitle='Multiplicity', var='# Sim Clusters', xtitle='Energy [GeV]'),
        'SimClustersEn_Eta': dict(ytitle=r'$\eta$', var='# Sim Clusters', xtitle='Energy [GeV]'),
        'SimClustersEn_Phi': dict(ytitle=r'$\phi$', var='# Sim Clusters', xtitle='Energy [GeV]'),
        'SimClustersMult_Eta': dict(xtitle='Multiplicity', var='# Sim Clusters', ytitle=r'$\eta$'),
        'RecoClustersEn_Mult': dict(ytitle='Multiplicity', var='# Reco Clusters', xtitle='Energy [GeV]'),
        'RecoClustersEn_Eta': dict(ytitle=r'$\eta$', var='# Reco Clusters', xtitle='Energy [GeV]', logz=True),
        'RecoClustersEn_Phi': dict(ytitle=r'$\phi$', var='# Reco Clusters', xtitle='Energy [GeV]', logz=True),
        'RecoClustersMult_Eta': dict(xtitle='Multiplicity', var='# Reco Clusters', ytitle=r'$\eta$'),
        'simToRecoScore_En': dict(ytitle='SimCluster Energy [GeV]', var='# Clusters', xtitle='SimToReco Score'),
        'simToRecoScore_EnFrac': dict(ytitle='Energy Fraction', var='# Clusters', xtitle='SimToReco Score'),
        'simToRecoScore_EnSimTrack': dict(ytitle='SimTrack Energy [GeV]', var='# Clusters', xtitle='SimToReco Score'),
        'simToRecoScore_Mult': dict(ytitle='Multiplicity', var='# Clusters', xtitle='SimToReco Score'),
        'simToRecoShEnF_En': dict(ytitle='SimCluster Energy [GeV]', var='# Clusters', xtitle='SimToReco Shared Energy Fraction'),
        'simToRecoShEnF_EnFrac': dict(ytitle='Energy Fraction', var='# Clusters', xtitle='SimToReco Shared Energy Fraction'),
        'simToRecoShEnF_EnSimTrack': dict(ytitle='SimTrack Energy [GeV]', var='# Clusters', xtitle='SimToReco Shared Energy Fraction'),
        'simToRecoShEnF_Mult': dict(ytitle='Multiplicity', var='# Clusters', xtitle='SimToReco Shared Energy Fraction'),
        'simToRecoShEnF_Score': dict(ytitle='SimToReco Score', var='# Clusters', xtitle='SimToReco Shared Energy Fraction', logz=True),
    }

    vars2D.update({
        **{f'{subdir}/ResponseE_Eta': dict(ytitle=titles['response'], var='# Clusters', xtitle=r'$\eta$') for subdir in subdirs},
        **{f'{subdir}/ResponseE_Phi': dict(ytitle=titles['response'], var='# Clusters', xtitle=r'$\phi$') for subdir in subdirs},
    })

    for name, props in vars2D.items():
        root_hist = cached_histos[f"{name}"]
        plot2D(root_hist, args.sample_label, args.era, props, outname=os.path.join(args.odir, name))

    vars1D = {
        'simToRecoShEnF': dict(xtitle='SimToReco Shared Energy Fraction', ytitle='# SimClusters', var='SimClusters', logy=False),
        'simToRecoScore': dict(xtitle='SimToReco Score', ytitle='# SimClusters', var='SimClusters', logy=False),
        'recoToSimScore': dict(xtitle='RecoToSim Score', ytitle='# SimClusters', var='SimClusters', logy=False),
    }

    for name, props in vars1D.items():
        root_hist = cached_histos[f"{name}"]
        plot1D(root_hist, args.sample_label, args.era, props, outname=os.path.join(args.odir, name))

    ##################################################################################
    # Temporary hack to access CaloParticle histrograms
    ##################################################################################

    if args.ticl:   sub_folder = 'TiclBarrel'
    else:           sub_folder = 'ParticleFlow'
    dqm_dir = f"DQMData/Run 1/HLT/Run summary/{sub_folder}/{matching}/CaloParticles"
    afile = ROOT.TFile.Open(args.file)
    utils.checkRootDir(afile, dqm_dir)

    debug('Start caching PFCluster histograms...')
    subdirs = []
    cached_histos = {}
    directory = afile.GetDirectory(dqm_dir)
    for key in directory.GetListOfKeys():
        obj = key.ReadObj()
        if isinstance(obj, ROOT.TDirectory):
            subdirs.append(obj.GetName())
            subdir = afile.GetDirectory(f"{dqm_dir}/{obj.GetName()}")
            for subkey in subdir.GetListOfKeys():
                name = subkey.GetName()
                cached_histos[f"{obj.GetName()}/{name}"] = subkey.ReadObj()
        else:
            name = key.GetName()
            cached_histos[name] = key.ReadObj()
    debug(' ...done.')

    vars1D = {
        'CP_simToRecoShEnF': dict(xtitle='SimToReco Shared Energy Fraction', ytitle='# CaloParticles', var='CaloParticles', logy=False),
        'CP_simToRecoScore': dict(xtitle='SimToReco Score', ytitle='# CaloParticles', var='CaloParticles', logy=False),
        'CP_recoToSimScore': dict(xtitle='RecoToSim Score', ytitle='# CaloParticles', var='CaloParticles', logy=False),
    }

    for name, props in vars1D.items():
        root_hist = cached_histos[f"{name}"]
        plot1D(root_hist, args.sample_label, args.era, props, outname=os.path.join(args.odir, name))
    
    vars2D = {
        'CP_simToRecoShEnF_Score': dict(ytitle='SimToReco Score', var='# CaloParticles', xtitle='SimToReco Shared Energy Fraction', logz=True),
    }

    for name, props in vars2D.items():
        root_hist = cached_histos[f"{name}"]
        plot2D(root_hist, args.sample_label, args.era, props, outname=os.path.join(args.odir, name))
