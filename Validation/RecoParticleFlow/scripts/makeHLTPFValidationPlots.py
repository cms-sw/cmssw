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

def createDir(adir):
    if not os.path.exists(adir):
        os.makedirs(adir)
    return adir

def checkRootDir(afile, adir):
    if not afile.Get(adir):
        raise RuntimeError(f"Directory '{adir}' not found in {afile}")
    
def GetRootSubDir(afile, adir):
    subdirs = []
    d = afile.GetDirectory(adir)
    for key in d.GetListOfKeys():
        obj = key.ReadObj()
        if isinstance(obj, ROOT.TDirectory):
            subdirs.append(obj.GetName())
    return subdirs

def checkRootFile(afile, hname, rebin=None):
    hist_orig = afile.Get(hname)
    if not hist_orig:
        raise RuntimeError(f"WARNING: Histogram {hname} not found.")

    if rebin is not None:
        hist = hist_orig.Clone(hname + "_clone")
        hist.SetDirectory(0) # detach from file

        if isinstance(rebin, (int, float)):
            hist = hist.Rebin(int(rebin), hname + "_rebin")
        elif hasattr(rebin, '__iter__'):
            bin_edges_c = array.array('d', rebin)
            hist = hist.Rebin(len(bin_edges_c) - 1, hname + "_rebin", bin_edges_c)
        else:
            raise ValueError(f"Unknown type for rebin: {type(rebin)}")

    else:
        hist = hist_orig

    return hist


def rate_errorbar_declutter(eff, err, yaxmin, frac=0.01):
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
    return eff_filt, (err_lo/2,err_hi/2), up_error, transform

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
            print(" ### INFO: Saving " + name + '.' + ext)
            plt.savefig(name + '.' + ext)
        plt.close()

def plotOverlay(subdirs, adir, name, props, outdir):
    """
    Plots 1D distributions, overlaying plots with identical names in different 'subdirs'.
    """
    colors_iter = iter(('#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
                        '#984ea3', '#999999', '#e41a1c', '#dede00')) #colour-blind friendly
    pattern = r"Score(\d+)p(\d+)"
    replacement = lambda m: f"score = {m.group(1)}.{m.group(2)}"
    
    plotter = Plotter(args.sample_label, grid_color=None)
    for sub in subdirs:
        root_hist = checkRootFile(afile, f"{adir}/{sub}/{name}", rebin=None)
        nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
        values, errors = histo_values_errors(root_hist)
        errors /= 2
    
        line = plotter.ax.stairs(values, bin_edges, linewidth=2,
                                 baseline=None, color=next(colors_iter))

        sublabel = re.sub(pattern, replacement, sub)        
        plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                            color=line.get_edgecolor(),
                            fmt='s', label=sublabel, **errorbar_kwargs)

    # plotter.limits_with_margin(valuesList, errorsList, logY=props['logy'])
    plotter.labels(x=props['xtitle'], y=props['ytitle'], legend_title='')    
    plotter.ax.grid(color='grey', axis='x')    
    plt.tight_layout()
    plotter.save( os.path.join(outdir, name) )

def plotOverlayRatio(subdirs, adir, num, den, props, outdir):
    """
    Plots 1D distributions of numerator / denominator.
    """
    colors_iter = iter(('#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
                        '#984ea3', '#999999', '#e41a1c', '#dede00')) #colour-blind friendly
    pattern = r"Score(\d+)p(\d+)"
    replacement = lambda m: f"score = {m.group(1)}.{m.group(2)}"
    
    plotter = Plotter(args.sample_label, grid_color=None)
    for sub in subdirs:
        hist_num = checkRootFile(afile, f"{adir}/{sub}/{num}", rebin=None)
        hist_den = checkRootFile(afile, f"{adir}/{sub}/{den}", rebin=None)
        
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

def plotEffComp1D(cached_histos, vars1d, outdir, text, top_text=False, suffix=''):
    """
    Plots 1D distributions.
    The `avars` variables is a dictionary whose values are (xlabel, ylabel, rebin).
    """
    plotter = Plotter(args.sample_label, grid_color=None)
    ax2 = plotter.ax.twinx()
    eff_color = '#bd1f01'
    ax2.set_ylabel('Efficiency', color=eff_color)

    valuesList, errorsList = [], []
    colors_iter = iter(('black', 'blue'))
    for avar in vars1d:
        name, (xlabel, ylabel, rebin, logy, doNormalize, leglabel) = avar
        if doNormalize:
            ylabel = '[a.u.]'

        root_hist = cached_histos[name]
        nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
        values, errors = histo_values_errors(root_hist)
        errors /= 2

        # normalization
        doNormalize = False
        if 'Eff' not in name and doNormalize:
            errors /= sum(values)
            values /= sum(values)

        if 'Eff' in name:
            # ax2.set_yscale('log')
            ax2.set_ylim(-0.05, 1.15)

            # eff_filt, (err_filt_lo, err_filt_hi), up_error, transform = rate_errorbar_declutter(eff_values, eff_errors, axmin)
            ax2.stairs(values, bin_edges, linewidth=2, baseline=None, color=eff_color)
            ax2.errorbar(bin_centers, values, xerr=None, yerr=errors,
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
    plotter.save( os.path.join(outdir, name) )

def plot1Dvars(afile, adir, avars, outdir, text, top_text=False):
    for var, (xlabel, ylabel, rebin, logy, _) in avars.items():
        plotter = Plotter(args.sample_label)
        root_hist = checkRootFile(afile, f"{adir}/{var}", rebin=rebin)
        nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
        values, errors = histo_values_errors(root_hist)
        errors /= 2

        plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                            fmt='s', color='black', label=var, **errorbar_kwargs)
        plotter.ax.stairs(values, bin_edges, color='black', linewidth=2, baseline=None)

        plotter.ax.text(0.03, 0.97, text, transform=plotter.ax.transAxes, fontsize=fontsize,
                        verticalalignment='top', horizontalalignment='left')

        plotter.limits_with_margin(values, errors, logY=logy)
        plotter.labels(x=xlabel, y=ylabel)

        if top_text:
            plotter.ax.text(0.97, 0.97, root_hist.GetTitle().replace('ET', r'$E_T$'),
                            transform=plotter.ax.transAxes, fontsize=fontsize,
                            verticalalignment='top', horizontalalignment='right')

        plt.tight_layout()
        plotter.save( os.path.join(outdir, var, suffix) )


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
    parser.add_argument('-l', '--sample_label', default="QCD (200 PU)", help='Sample label for plotting.')

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

    nEventsLabel = '# Events'
    effLabel = 'Efficiency'

    dqm_dir = f"DQMData/Run 1/HLT/Run summary/ParticleFlow/PFClusterValidation"
    afile = ROOT.TFile.Open(args.file)
    checkRootDir(afile, dqm_dir)

    print("### INFO: Start caching histograms...")
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
    print("...done.")

    # subdirs = GetRootSubDir(afile, dqm_dir)
    for subdir in subdirs:
        checkRootDir(afile, f"{dqm_dir}/{subdir}")
        createDir(f'{args.odir}/{subdir}')
        for name, suf in zip(('', '_Reconstructable'), ('', 'Reconstructable')):
            vars1D = {
                # PF tester producer
                f'SimClusters{suf}En': ('Energy [GeV]', nEventsLabel, None, True, False, 'Sim'),
                f'{subdir}/SimClustersMatchedRecoClustersEn': ('Energy [GeV]', nEventsLabel, None, True, False, 'Reco'),
                f'{subdir}/Eff_vs_Energy{name}': ('Energy [GeV]', nEventsLabel, None, True, False, None),
                f'SimClusters{suf}Pt': (r'$p_{T}$ [GeV]', nEventsLabel, None, True, False, 'Sim'),
                f'{subdir}/SimClustersMatchedRecoClustersPt': (r'$p_{T}$ [GeV]', nEventsLabel, None, True, False, 'Reco'),
                f'{subdir}/Eff_vs_Pt{name}': (r'$p_{T}$ [GeV]', nEventsLabel, None, True, False, None),
                f'SimClusters{suf}Eta': (r'$\eta$', nEventsLabel, None, False, False, 'Sim'),
                f'{subdir}/SimClustersMatchedRecoClustersEta': (r'$\eta$', nEventsLabel, None, False, False, 'Reco'),
                f'{subdir}/Eff_vs_Eta{name}': (r'$\eta$', nEventsLabel, None, False, False, None),
                f'SimClusters{suf}Phi': (r'$\phi$', nEventsLabel, None, False, False, 'Sim'),
                f'{subdir}/SimClustersMatchedRecoClustersPhi': (r'$\phi$', nEventsLabel, None, False, False, 'Reco'),
                f'{subdir}/Eff_vs_Phi{name}': (r'$\phi$', nEventsLabel, None, False, False, None),
                f'SimClusters{suf}Mult': ('Multiplicity', nEventsLabel, None, True, False, 'Sim'),
                f'{subdir}/SimClustersMatchedRecoClustersMult': ('Multiplicity', nEventsLabel, None, True, False, 'Reco'),
                f'{subdir}/Eff_vs_Mult{name}': ('Multiplicity', nEventsLabel, None, True, False, None),
            }

            # Compare pairs of variables
            it = iter(vars1D.items())
            for var in it:
                avars = (var, next(it), next(it)) # reco, sim and efficiency for a given variable
                plotEffComp1D(cached_histos, vars1d=avars, outdir=args.odir, text='', suffix=f'')
                
    titles = {'response': r"$<p_{T}^{Reco}/p_{T}^{Sim}>$", 'resolution': r"$\sigma(p_{T}^{Reco}/p_{T}^{Sim}) / <p_{T}^{Reco}/p_{T}^{Sim}>$", 'eff': 'Efficiency'}

    varsOverlay = {
        "ResponsePt_Mean"             : dict(ytitle=titles['response'], xtitle=r'$p_{T} [GeV]$', logy=False),
        "ResponseEta_Mean"            : dict(ytitle=titles['response'], xtitle=r'$\eta$', logy=False),
        "ResponsePhi_Mean"            : dict(ytitle=titles['response'], xtitle=r'$\phi$', logy=False),
        "ResponseMult_Mean"           : dict(ytitle=titles['response'], xtitle='Multiplicity', logy=False),
        "Eff_vs_Pt_Reconstructable"   : dict(ytitle=titles['eff'], xtitle='$p_{T} [GeV]$', logy=False),
        "Eff_vs_Eta_Reconstructable"  : dict(ytitle=titles['eff'], xtitle=r'$\eta$', logy=False),
        "Eff_vs_Phi_Reconstructable"  : dict(ytitle=titles['eff'], xtitle=r'$\phi$', logy=False),
        "Eff_vs_Mult_Reconstructable" : dict(ytitle=titles['eff'], xtitle='Multiplicity', logy=False),
        }
    for name, props in varsOverlay.items():
        plotOverlay(subdirs, dqm_dir, name, props, outdir=args.odir)

    varsResponse = {
        ("ResponsePt_Sigma", "ResponsePt_Mean")     : dict(name='ResolutionPt', ytitle=titles['resolution'], xtitle=r'$p_{T} [GeV]$', logy=False),
        ("ResponseEta_Sigma", "ResponseEta_Mean")   : dict(name='ResolutionEta', ytitle=titles['resolution'], xtitle=r'$\eta$', logy=False),
        ("ResponsePhi_Sigma", "ResponsePhi_Mean")   : dict(name='ResolutionPhi',ytitle=titles['resolution'], xtitle=r'$\phi$', logy=False),
        ("ResponseMult_Sigma", "ResponseMult_Mean") : dict(name='ResolutionMult',ytitle=titles['resolution'], xtitle='Multiplicity', logy=False),
    }
    for (num, den), props in varsResponse.items():
        plotOverlayRatio(subdirs, dqm_dir, num, den, props, outdir=args.odir)
