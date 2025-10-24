#!/usr/bin/env python3

import os
import argparse
import numpy as np
import hist
import re
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
plt.style.use('tableau-colorblind10')
import array
import ROOT
import mplhep as hep
hep.style.use("CMS")

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def debug(mes):
    print('### INFO: ' + mes)
    
def createDir(adir):
    if not os.path.exists(adir):
        os.makedirs(adir)
    return adir

def createIndexPHP(src, dest):
    php_file = os.path.join(src, 'index.php')
    if os.path.exists(php_file):
        os.system(f'cp {php_file} {dest}')

def checkRootDir(afile, adir):
    if not afile.Get(adir):
        raise RuntimeError(f"Directory '{adir}' not found in {afile}")

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

class Plotter:
    def __init__(self, label, fontsize=18, grid_color='grey'):
        self._fig, self._ax = plt.subplots(figsize=(10, 10))
        self.fontsize = fontsize
        
        hep.cms.text(' Phase-2 Simulation Preliminary', ax=self._ax, fontsize=fontsize)
        hep.cms.lumitext(label + " | 14 TeV", ax=self._ax, fontsize=fontsize)
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

def plotProject(h, props, rebin_edges, outname):
    """
    Project and plot slices of a 2D histogram.
    """
    colors_iter = iter(('#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
                        '#984ea3', '#999999', '#e41a1c', '#dede00')) #colour-blind friendly

    valuesList, errorsList = [], []
    plotter = Plotter(args.sample_label, grid_color=None)
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
        label = f"{low} < {props['var']} < {high} {props['unit']}"
        plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                            color=line.get_edgecolor(),
                            fmt='s', label=label, **errorbar_kwargs)

    plotter.limits_with_margin(valuesList, errorsList, logY=props['logy'])
    
    plotter.labels(x=props['xtitle'], y=props['ytitle'], legend_title='')    
    plotter.ax.grid(color='grey', axis='x')    
    plt.tight_layout()
    plotter.save(outname)

def plotOverlay(subdirs, cached_histos, name, props, outdir):
    """
    Plots 1D distributions, overlaying plots with identical names in different 'subdirs'.
    """
    colors_iter = iter(('#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
                        '#984ea3', '#999999', '#e41a1c', '#dede00')) #colour-blind friendly
    pattern = r"Score(\d+)p(\d+)"
    replacement = lambda m: f"score = {m.group(1)}.{m.group(2)}"
    
    plotter = Plotter(args.sample_label, grid_color=None)
    for sub in subdirs:
        root_hist = cached_histos[f"{sub}/{name}"]

        if props['rebin'] is not None:
            root_hist = root_hist.Clone(f"{name}" + "_clone")
            root_hist.SetDirectory(0) # detach from file

            if isinstance(props['rebin'], (int, float)):
                root_hist = root_hist.Rebin(int(props['rebin']), f"{name}" + "_rebin")
            elif hasattr(props['rebin'], '__iter__'):
                bin_edges_c = array.array('d', props['rebin'])
                root_hist = root_hist.Rebin(len(bin_edges_c) - 1, f"{name}" + "_rebin", bin_edges_c)
            else:
                raise ValueError(f"Unknown type for rebin: {type(props['rebin'])}")

        nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
        values, errors = histo_values_errors(root_hist)
        errors /= 2
    
        line = plotter.ax.stairs(values, bin_edges, linewidth=2,
                                 baseline=None, color=next(colors_iter))

        sublabel = re.sub(pattern, replacement, sub)
        if sublabel == 'score = 1.1': sublabel = 'score = 1.0'
        eff_filt, (err_filt_lo, err_filt_hi), up_error, transform = rate_errorbar_declutter(plotter, values, errors, 0)
        plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=[err_filt_lo,err_filt_hi],
                            color=line.get_edgecolor(),
                            fmt='s', label=sublabel, **errorbar_kwargs)

    plotter.limits(y=(0,1.1))
    plotter.labels(x=props['xtitle'], y=props['ytitle'], legend_title='')    
    plotter.ax.grid(color='grey', axis='x')    
    plt.tight_layout()
    plotter.save( os.path.join(outdir, name) )

def plotOverlayRatio(subdirs, cached_histos, num, den, props, outdir):
    """
    Plots 1D distributions of numerator / denominator.
    """
    colors_iter = iter(('#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
                        '#984ea3', '#999999', '#e41a1c', '#dede00')) #colour-blind friendly
    pattern = r"Score(\d+)p(\d+)"
    replacement = lambda m: f"score = {m.group(1)}.{m.group(2)}"
    
    plotter = Plotter(args.sample_label, grid_color=None)
    for sub in subdirs:
        hist_num = cached_histos[f"{sub}/{num}"]
        hist_den = cached_histos[f"{sub}/{den}"]

        if props['rebin'] is not None:
            hist_num = hist_num.Clone(f"{sub}/{num}" + "_clone")
            hist_den = hist_den.Clone(f"{sub}/{den}" + "_clone")
            hist_num.SetDirectory(0) # detach from file
            hist_den.SetDirectory(0) # detach from file

            if isinstance(props['rebin'], (int, float)):
                hist_num = hist_num.Rebin(int(props['rebin']), f"{sub}/{num}" + "_rebin")
                hist_den = hist_den.Rebin(int(props['rebin']), f"{sub}/{den}" + "_rebin")
            elif hasattr(props['rebin'], '__iter__'):
                bin_edges_c = array.array('d', props['rebin'])
                hist_num = hist_num.Rebin(len(bin_edges_c) - 1, f"{sub}/{num}" + "_rebin", bin_edges_c)
                hist_den = hist_den.Rebin(len(bin_edges_c) - 1, f"{sub}/{den}" + "_rebin", bin_edges_c)
            else:
                raise ValueError(f"Unknown type for rebin: {type(props['rebin'])}")
        
        nbins, bin_edges, bin_centers, bin_widths = define_bins(hist_num)
        num_vals, num_errors = histo_values_errors(hist_num)
        den_vals, den_errors = histo_values_errors(hist_den)

        ratio_vals = [s / m if m != 0 else np.nan for s, m in zip(num_vals, den_vals)]
        ratio_errors = [np.sqrt((ds / m)**2 + ((s / m) * (dm / m))**2) / 2 if m != 0 else np.nan
                        for s, ds, m, dm in zip(num_vals, num_errors, den_vals, den_errors)]

        line = plotter.ax.stairs(ratio_vals, bin_edges, linewidth=2,
                                 baseline=None, color=next(colors_iter))

        sublabel = re.sub(pattern, replacement, sub)        
        plotter.ax.errorbar(bin_centers, ratio_vals, xerr=None, yerr=ratio_errors,
                            color=line.get_edgecolor(),
                            fmt='s', label=sublabel, **errorbar_kwargs)

    # plotter.limits_with_margin(valuesList, errorsList, logY=props['logy'])
    plotter.labels(x=props['xtitle'], y=props['ytitle'], legend_title='')    
    plotter.ax.grid(color='grey', axis='x')    
    plt.tight_layout()
    plotter.save( os.path.join(outdir, props['name']) )

def plotEffComp1D(cached_histos, title, vars1d, outdir, text, top_text=False, suffix=''):
    """
    Plots 1D distributions.
    """
    plotter = Plotter(args.sample_label, grid_color=None)
    ax2 = plotter.ax.twinx()
    eff_color = '#bd1f01'
    
    valuesList, errorsList = [], []
    colors_iter = iter(('black', 'blue'))

    histo_names = [vars1d['den'], vars1d['num'], vars1d['ratio']]
    leg_names = [vars1d['legden'], vars1d['legnum'], '']
    rebin = vars1d['rebin']
    doNormalize = vars1d['normalize']
    logy = vars1d['logy']
    xlabel = vars1d['xtitle']
    ylabel = vars1d['ytitle']

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
        if 'Eff' not in name and doNormalize:
            errors /= sum(values)
            values /= sum(values)
            
        if any(x in name for x in ('Eff', 'Fake', 'Duplicate', 'Merge')):
            if 'Eff' in name:
                axis_name = 'Efficiency'
            elif 'Fake' in name:
                axis_name = 'Fake Rate'
            elif 'Dup' in name:
                axis_name = 'Duplicate Rate'
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

def plot2D(h, props, outname):
    """
    Plot with mplhep's hist2d (preserves ROOT bin edges, color bar included)
    empty bins will be invisible (background color).
    """
    plotter = Plotter(args.sample_label, fontsize=15)

    nbins_x, nbins_y, x_edges, y_edges = define_bins_2D(h)
    values = histo_values_2D(h)
    
    pcm = plotter.ax.pcolormesh(x_edges, y_edges, np.where(values==0, np.nan, values),
                                cmap='viridis', shading='auto')

    plotter.labels(x=props['xtitle'], y=props['ytitle'])
    plotter.fig.colorbar(pcm, ax=plotter.ax, label='# Clusters')
    plt.tight_layout()
    plotter.save(outname)
        
# def plot1D(afile, adir, avars, outdir, text, top_text=False):
#     for var, (xlabel, ylabel, rebin, logy, _) in avars.items():
#         plotter = Plotter(args.sample_label)
#         root_hist = checkRootFile(afile, f"{adir}/{var}", rebin=rebin)
#         nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
#         values, errors = histo_values_errors(root_hist)
#         errors /= 2

#         plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
#                             fmt='s', color='black', label=var, **errorbar_kwargs)
#         plotter.ax.stairs(values, bin_edges, color='black', linewidth=2, baseline=None)

#         plotter.ax.text(0.03, 0.97, text, transform=plotter.ax.transAxes, fontsize=fontsize,
#                         verticalalignment='top', horizontalalignment='left')

#         plotter.limits_with_margin(values, errors, logY=logy)
#         plotter.labels(x=xlabel, y=ylabel)

#         if top_text:
#             plotter.ax.text(0.97, 0.97, root_hist.GetTitle().replace('ET', r'$E_T$'),
#                             transform=plotter.ax.transAxes, fontsize=fontsize,
#                             verticalalignment='top', horizontalalignment='right')

#         plt.tight_layout()
#         plotter.save( os.path.join(outdir, var) )


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
    parser.add_argument('--EnFracCut', default=0.01, help='Cut on the sim cluster energy fraction.')
    parser.add_argument('--PtCut', default=0.01, help='Cut on the sim cluster energy fraction.')

    mutual_excl2 = parser.add_mutually_exclusive_group(required=True)
    mutual_excl2.add_argument('-f', '--file', help='Paths to the DQM ROOT file.')
    mutual_excl2.add_argument('-x', '--compare_files', nargs='+', type=check_list_length,
                              help='Compare the same collection in different DQM files.', )
    parser.add_argument('-y', '--compare_files_labels', nargs='+',
                        action=DependencyAction, help='Compare the same collection in different DQM files.',)
    
    args = parser.parse_args()

    createDir(args.odir)
    
    fontsize = 16    
    colors = hep.style.CMS['axes.prop_cycle'].by_key()['color']
    markers = ('o', 's', 'd')
    errorbar_kwargs = dict(capsize=3, elinewidth=0.8, capthick=2, linewidth=2, linestyle='')

    nSimClustersLabel = '# SimClusters'
    nPFClustersLabel = '# PFClusters'
    effLabel = 'Efficiency'

    dqm_dir = f"DQMData/Run 1/HLT/Run summary/ParticleFlow/PFClusterValidation_EnFracCut{str(args.EnFracCut).replace('.', 'p')}_PtCut{str(args.PtCut).replace('.', 'p')}"
    afile = ROOT.TFile.Open(args.file)
    checkRootDir(afile, dqm_dir)

    debug('Start caching histograms...')
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

    for subdir in subdirs:
        checkRootDir(afile, f"{dqm_dir}/{subdir}")
        createDir(f'{args.odir}/{subdir}')
        createIndexPHP(src=args.odir, dest=f'{args.odir}/{subdir}')
        
        for name, suf in zip(('', '_Reconstructable'), ('', 'Reconstructable')):
            varsDict = {
                # Cluster efficiency
                f'{subdir}/Eff_vs_En{name}': dict(ratio=f'{subdir}/Eff_vs_En{name}', 
                    den=f'SimClusters{suf}En', legden='SimClusters',
                    num=f'{subdir}/SimClustersMatchedRecoClustersEn', legnum='Matched SimClusters',
                    xtitle='Energy from SimTrack [GeV]', ytitle=nSimClustersLabel, rebin=4, logy=False, normalize=False),
                f'{subdir}/Eff_vs_EnHits{name}': dict(ratio=f'{subdir}/Eff_vs_EnHits{name}', 
                    den=f'SimClusters{suf}EnHits', legden='SimClusters',
                    num=f'{subdir}/SimClustersMatchedRecoClustersEnHits', legnum='Matched SimClusters', 
                    xtitle='Energy from hits [GeV]', ytitle=nSimClustersLabel, rebin=4, logy=False, normalize=False),
                f'{subdir}/Eff_vs_EnFrac{name}': dict(ratio=f'{subdir}/Eff_vs_EnFrac{name}', 
                    den=f'SimClusters{suf}EnFrac', legden='SimClusters',
                    num=f'{subdir}/SimClustersMatchedRecoClustersEnFrac', legnum='Matched SimClusters', 
                    xtitle='Energy Fraction', ytitle=nSimClustersLabel, rebin=4, logy=False, normalize=False),
                f'{subdir}/Eff_vs_Pt{name}': dict(ratio=f'{subdir}/Eff_vs_Pt{name}', 
                    den=f'SimClusters{suf}Pt', legden='SimClusters',
                    num=f'{subdir}/SimClustersMatchedRecoClustersPt', legnum='Matched SimClusters', 
                    xtitle=r'$p_{T}$ [GeV]', ytitle=nSimClustersLabel, rebin=4, logy=False, normalize=False),
                f'{subdir}/Eff_vs_Eta{name}': dict(ratio=f'{subdir}/Eff_vs_Eta{name}', 
                    den=f'SimClusters{suf}Eta', legden='SimClusters',
                    num=f'{subdir}/SimClustersMatchedRecoClustersEta', legnum='Matched SimClusters', 
                    xtitle=r'$\eta$', ytitle=nSimClustersLabel, rebin=None, logy=False, normalize=False),
                f'{subdir}/Eff_vs_Phi{name}': dict(ratio=f'{subdir}/Eff_vs_Phi{name}', 
                    den=f'SimClusters{suf}Phi', legden='SimClusters',
                    num=f'{subdir}/SimClustersMatchedRecoClustersPhi', legnum='Matched SimClusters', 
                    xtitle=r'$\phi$', ytitle=nSimClustersLabel, rebin=None, logy=False, normalize=False),
                # Cluster fake rate
                f'{subdir}/Fake_vs_En{name}': dict(ratio=f'{subdir}/Fake_vs_En{name}', 
                    den=f'RecoClusters{suf}En', legden='RecoClusters',
                    num=f'{subdir}/RecoClustersMatchedSimClustersEn', legnum='Matched RecoClusters', 
                    xtitle='Energy from SimTrack [GeV]', ytitle=nPFClustersLabel, rebin=4, logy=False, normalize=False),
                f'{subdir}/Fake_vs_Pt{name}': dict(ratio=f'{subdir}/Fake_vs_Pt{name}', 
                    den=f'RecoClusters{suf}Pt', legden='RecoClusters',
                    num=f'{subdir}/RecoClustersMatchedSimClustersPt', legnum='Matched RecoClusters', 
                    xtitle=r'$p_{T}$ [GeV]', ytitle=nPFClustersLabel, rebin=4, logy=False, normalize=False),
                f'{subdir}/Fake_vs_Eta{name}': dict(ratio=f'{subdir}/Fake_vs_Eta{name}', 
                    den=f'RecoClusters{suf}Eta', legden='RecoClusters',
                    num=f'{subdir}/RecoClustersMatchedSimClustersEta', legnum='Matched RecoClusters', 
                    xtitle=r'$\eta$', ytitle=nPFClustersLabel, rebin=None, logy=False, normalize=False),
                f'{subdir}/Fake_vs_Phi{name}': dict(ratio=f'{subdir}/Fake_vs_Phi{name}', 
                    den=f'RecoClusters{suf}Phi', legden='RecoClusters',
                    num=f'{subdir}/RecoClustersMatchedSimClustersPhi', legnum='Matched RecoClusters', 
                    xtitle=r'$\phi$', ytitle=nPFClustersLabel, rebin=None, logy=False, normalize=False),
                f'{subdir}/Fake_vs_Mult{name}': dict(ratio=f'{subdir}/Fake_vs_Mult{name}', 
                    den=f'RecoClusters{suf}Mult', legden='RecoClusters',
                    num=f'{subdir}/RecoClustersMatchedSimClustersMult', legnum='Matched RecoClusters', 
                    xtitle='Multiplicity', ytitle=nPFClustersLabel, rebin=4, logy=False, normalize=False),
            }

            # Compare pairs of variables
            for title, props in varsDict.items():
                plotEffComp1D(cached_histos, title, vars1d=props, outdir=args.odir, text='', suffix=f'')
                
    titles = {'response': r"$p_{T}^{Reco}/p_{T}^{Sim}$", 
              'av_response': r"$<p_{T}^{Reco}/p_{T}^{Sim}>$", 
              'resolution': r"$\sigma(p_{T}^{Reco}/p_{T}^{Sim}) / <p_{T}^{Reco}/p_{T}^{Sim}>$", 
              'eff': 'Efficiency'}

    varsOverlay = {
        "ResponseEn_Mean"               : dict(ytitle=titles['response'], rebin=None, xtitle='Energy from SimTrack [GeV]', logy=False),
        "ResponseEnHits_Mean"           : dict(ytitle=titles['response'], rebin=None, xtitle='Energy from hits [GeV]', logy=False),
        "ResponseEnFrac_Mean"           : dict(ytitle=titles['response'], rebin=None, xtitle='Energy Fraction', logy=False),
        "ResponsePt_Mean"               : dict(ytitle=titles['response'], rebin=None, xtitle=r'$p_{T} [GeV]$', logy=False),
        "ResponseEta_Mean"              : dict(ytitle=titles['response'], rebin=None, xtitle=r'$\eta$', logy=False),
        "ResponsePhi_Mean"              : dict(ytitle=titles['response'], rebin=None, xtitle=r'$\phi$', logy=False),
        "ResponseMult_Mean"             : dict(ytitle=titles['response'], rebin=None, xtitle='Multiplicity', logy=False),
        "Eff_vs_En_Reconstructable"     : dict(ytitle=titles['eff'], rebin=4, xtitle='Energy from SimTrack [GeV]', logy=False),
        "Eff_vs_EnHits_Reconstructable" : dict(ytitle=titles['eff'], rebin=4, xtitle='Energy from hits [GeV]', logy=False),
        "Eff_vs_EnFrac_Reconstructable" : dict(ytitle=titles['eff'], rebin=6, xtitle='Energy Fraction', logy=False),
        "Eff_vs_Pt_Reconstructable"     : dict(ytitle=titles['eff'], rebin=4, xtitle='$p_{T} [GeV]$', logy=False),
        "Eff_vs_Eta"                    : dict(ytitle=titles['eff'], rebin=None, xtitle=r'$\eta$', logy=False),
        "Eff_vs_Eta_Reconstructable"    : dict(ytitle=titles['eff'], rebin=None, xtitle=r'$\eta$', logy=False),
        "Eff_vs_Phi_Reconstructable"    : dict(ytitle=titles['eff'], rebin=None, xtitle=r'$\phi$', logy=False),
        "Eff_vs_Mult_Reconstructable"   : dict(ytitle=titles['eff'], rebin=4, xtitle='Multiplicity', logy=False),
        }
    for name, props in varsOverlay.items():
        plotOverlay(subdirs, cached_histos, name, props, outdir=args.odir)

    varsResponse = {
        ("ResponseEn_Sigma", "ResponseEn_Mean")     : dict(name='ResolutionEn', ytitle=titles['resolution'], xtitle=r'$E [GeV]$', rebin=4, logy=False),
        ("ResponseEnHits_Sigma", "ResponseEnHits_Mean") : dict(name='ResolutionEnHits', ytitle=titles['resolution'], xtitle=r'$E_{hits} [GeV]$', rebin=4, logy=False),
        ("ResponseEnFrac_Sigma", "ResponseEnFrac_Mean") : dict(name='ResolutionEnFrac', ytitle=titles['resolution'], xtitle=r'Energy Fraction', rebin=4, logy=False),
        ("ResponsePt_Sigma", "ResponsePt_Mean")     : dict(name='ResolutionPt', ytitle=titles['resolution'], xtitle=r'$p_{T} [GeV]$', rebin=4, logy=False),
        ("ResponseEta_Sigma", "ResponseEta_Mean")   : dict(name='ResolutionEta', ytitle=titles['resolution'], xtitle=r'$\eta$', rebin=4, logy=False),
        ("ResponsePhi_Sigma", "ResponsePhi_Mean")   : dict(name='ResolutionPhi',ytitle=titles['resolution'], xtitle=r'$\phi$', rebin=None, logy=False),
        ("ResponseMult_Sigma", "ResponseMult_Mean") : dict(name='ResolutionMult',ytitle=titles['resolution'], xtitle='Multiplicity', rebin=None, logy=False),
    }
    for (num, den), props in varsResponse.items():
        plotOverlayRatio(subdirs, cached_histos, num, den, props, outdir=args.odir)

    vars2D = {
        **{f'{subdir}/ResponseEn': dict(xtitle=titles['response'], ytitle='# Clusters', var=r'E', unit='[GeV]', logy=False, rebin=(0., 1., 3., 10., 100.)) for subdir in subdirs},
        **{f'{subdir}/ResponseEn': dict(xtitle=titles['response'], ytitle='# Clusters', var=r'E_{hist}', unit='[GeV]', logy=False, rebin=(0., 1., 3., 10., 100.)) for subdir in subdirs},
        **{f'{subdir}/ResponseEn': dict(xtitle=titles['response'], ytitle='# Clusters', var=r'Energy Fraction', unit='', logy=False, rebin=(0., 1., 3., 10., 100.)) for subdir in subdirs},
        **{f'{subdir}/ResponseEn': dict(xtitle=titles['response'], ytitle='# Clusters', var=r'$p_{T}$', unit='[GeV]', logy=False, rebin=(0., 1., 3., 10., 100.)) for subdir in subdirs},
        **{f'{subdir}/ResponseEn': dict(xtitle=titles['response'], ytitle='# Clusters', var=r'$\eta$', unit='', logy=False, rebin=(-1.5, -0.75, 0., 0.75, 1.5)) for subdir in subdirs},
        **{f'{subdir}/ResponseEn': dict(xtitle=titles['response'], ytitle='# Clusters', var=r'$\phi$', unit='', logy=False, rebin=(-3.15, -2., -1., 0., 1., 2., 3.15)) for subdir in subdirs},
        **{f'{subdir}/ResponseEn': dict(xtitle=titles['response'], ytitle='# Clusters', var='Multiplicity', unit='', logy=False, rebin=(0., 20., 45., 100., 200.)) for subdir in subdirs},
        'SimClustersReconstructableEnFrac_Mult': dict(xtitle='Multiplicity', ytitle='# Clusters', var='Energy Fraction', unit='', logy=True, rebin=(0., 0.005, 0.01, 0.02, 0.03, 1.)),
    }

    for name, props in vars2D.items():
        plotter = Plotter(args.sample_label, fontsize=15)
        root_hist = cached_histos[f"{name}"]
        plotProject(root_hist, props, rebin_edges=props['rebin'], outname=os.path.join(args.odir, name + '_Projected'))

        xtitle, ytitle, var = props['xtitle'], props['ytitle'], props['var']
        props['xtitle'] = var
        props['ytitle'] = xtitle
        # props['var'] = ytitle
        plot2D(root_hist, props, outname=os.path.join(args.odir, name))
