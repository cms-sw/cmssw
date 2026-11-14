#!/usr/bin/env python3

import os
import argparse
import numpy as np
import hist
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
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
    rebin: int = None
    out: str = ''
    logy: bool = False
    logz: bool = False
    drawdiag: bool = False

def createDir(adir):
    if not os.path.exists(adir):
        os.makedirs(adir)
    return adir

def checkRootDir(afile, adir):
    if not afile.Get(adir):
        raise RuntimeError(f"Directory '{adir}' not found in {afile}")

def checkRootFile(afile, hname, rebin=None):
    hist_orig = afile.Get(hname)
    if not hist_orig:
        raise RuntimeError(f"WARNING: Histogram {hname} not found.")

    hist = hist_orig.Clone(hname + "_clone")
    hist.SetDirectory(0) # detach from file

    if rebin is not None:
        if isinstance(rebin, (int, float)):
            hist = hist.Rebin(int(rebin), hname + "_rebin")
        elif hasattr(rebin, '__iter__'):
            bin_edges_c = array.array('d', rebin)
            hist = hist.Rebin(len(bin_edges_c) - 1, hname + "_rebin", bin_edges_c)
        else:
            raise ValueError(f"Unknown type for rebin: {type(rebin)}")

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

    def limits_with_margin(self, values, errors, logY=False):
        values[np.isinf(values) | np.isnan(values)] = 0.
        errors[np.isinf(errors) | np.isnan(errors)] = 0.
        
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

def plot1D(plotter, h, text, props):
    """Plot 1D distributions."""
    nbins, bin_edges, bin_centers, bin_widths = define_bins(h)
    values, errors = histo_values_errors(h)
    errors /= 2

    plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                        fmt='s', color='black', label=props.x, **errorbar_kwargs)
    plotter.ax.stairs(values, bin_edges, color='black', linewidth=2, baseline=None)
    plotter.ax.text(0.03, 0.97, text, transform=plotter.ax.transAxes, fontsize=16,
                    verticalalignment='top', horizontalalignment='left')

    plotter.limits_with_margin(values, errors, logY=props.logy)
    plotter.labels(x=props.x, y=props.y)
    return plotter

def plot2D(plotter, h, text, props):
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
        root_hist = checkRootFile(afile, f"{adir}/{var}", rebin=props.rebin)

        if mode == '1D':
            plotter = plot1D(plotter, root_hist, metType, props)
        elif mode == '2D':
            plotter = plot2D(plotter, root_hist, metType, props)
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

    for var, props in avars.items():
        plotter = Plotter(args.sample_label, period=args.period, en_com=args.energy)

        amax, amin = float('-inf'), float('+inf')
        for metType in metTypes:
            root_hist = checkRootFile(afile, f"{adir}/{metType}/{var}", rebin=props.rebin)
            nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
            values, errors = histo_values_errors(root_hist)
            errors /= 2 # symmetrize

            patch = plotter.ax.stairs(values, bin_edges, linewidth=2, baseline=None)
            plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                                fmt='s', label=metType, color=patch.get_edgecolor(),
                                **errorbar_kwargs)
            
            amax = max(amax, (values + errors).max())
            amin = min(amin, (values - errors).min())

        diff_step = 0.05 * abs(amax-amin)
        plotter.limits(y=(amin - diff_step, amax + 2*diff_step), logY=False)
        plotter.labels(x=props.x, y=props.y)
        plotter.ax.legend(prop={'size': 15})
        plotter.save( os.path.join(outdir, var) )

def plot1DFilesComparison(adir, avars, outdir, files, files_labels, metType):
    """
    Plots 1D distributions.
    The `avars` variables is a dictionary whose values are (xlabel, ylabel, rebin).
    """
    outdir = os.path.join(outdir, 'ComparisonFiles_' + '_'.join(files_labels), metType)
    createDir(outdir)

    for var, props in avars.items():
        plotter = Plotter(args.sample_label, period=args.period, en_com=args.energy)

        amax, amin = float('-inf'), float('+inf')
        for afile, alabel in zip(files, files_labels):
            afile = ROOT.TFile.Open(afile)
            root_hist = checkRootFile(afile, f"{adir}/{metType}/{var}", rebin=props.rebin)
            nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
            values, errors = histo_values_errors(root_hist)
            errors /= 2 # symmetrize

            patch = plotter.ax.stairs(values, bin_edges, linewidth=2, baseline=None)
            plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                                fmt='s', label=alabel, color=patch.get_edgecolor(),
                                **errorbar_kwargs)
            
            amax = max(amax, (values + errors).max())
            amin = min(amin, (values - errors).min())

        diff_step = 0.05 * abs(amax-amin)
        plotter.limits(y=(amin - diff_step, amax + 2*diff_step), logY=False)
        plotter.labels(x=props.x, y=props.y)
        plotter.ax.legend(prop={'size': 15})
        plotter.save( os.path.join(outdir, var) )

def varsToPlot(metColl):
    """
    Define which variables to plot and the corresponding plot properties.
    """
    vars1D = {
        # MET and MHT common variables
        'MET'                     : HLTMETInputs(x=r'$MET_\text{Reco}$ [GeV]', y=nEvts, rebin=2, logy=False),
        'GenMETTrue'              : HLTMETInputs(x=r'$MET_\text{Gen}$ [GeV]', y=nEvts, rebin=2),
        'MEx'                     : HLTMETInputs(x=r'$MET_x$ [GeV]', y=nEvts, rebin=4),
        'MEy'                     : HLTMETInputs(x=r'$MET_y$ [GeV]', y=nEvts, rebin=4),
        'METPhi'                  : HLTMETInputs(x=r'$\text{MET}\phi_\text{Reco}$', y=nEvts, rebin=2),
        'METDeltaPhi_GenMETTrue'  : HLTMETInputs(x=r'$\text{MET}\phi_\text{Reco} - \text{MET}\phi_\text{Gen}$', y=nEvts, rebin=2),
        'METDiff_GenMETTrue'      : HLTMETInputs(x=r'$\text{MET}_\text{Reco} - \text{MET}_\text{Gen}$ [GeV]', y=nEvts, rebin=10),
        'METSignPseudo'           : HLTMETInputs(x='MET / √(∑E_T) "Significance" (event-by-event)', y=nEvts), # \sqrt{} throws a warning
        'METSignReal'             : HLTMETInputs(x='MET Significance (Likelihood)', y=nEvts), # covariance matrix missing
        'MET_Nvtx'                : HLTMETInputs(x='Number of vertices (MET-weighted)', y=nEvts, rebin=10),
        'Nvertex'                 : HLTMETInputs(x='Number of vertices', y=nEvts, rebin=6),
        'SumET'                   : HLTMETInputs(x=r'$\sum E_T$ [GeV]', y=nEvts, rebin=4),
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
        'METSignDiffAggr_Phi'  : HLTMETInputs(x=r'$\text{MET}\phi_{\text{Gen}}$',
                                              y=r'$\langle\text{MET}_\text{Reco}\rangle / \sigma(\text{MET}_\text{Reco}) - \langle\text{MET}_\text{Gen}\rangle / \sigma(\text{MET}_\text{Gen})$'),
    }

    isMHT = 'MHT' in metColl or metColl.lower().endswith('mht')
    isCalo = metColl in ('hltMet', 'hltCaloMET')
    if not isMHT:
        vars1D.update({
            # MET-specific
            'GenMETCalo'              : HLTMETInputs(x=r'$MET^\text{Calo}_\text{Gen}$ [GeV]', y=nEvts, rebin=2),
            'METDeltaPhi_GenMETCalo'  : HLTMETInputs(x=r'$\text{MET}\phi_\text{Reco} - \text{MET}\phi^{Calo}_\text{Gen}$', y=nEvts, rebin=2),
            'METDiff_GenMETCalo'      : HLTMETInputs(x=r'$\text{MET}_\text{Reco} - \text{MET}^\text{Calo}_\text{Gen}$ [GeV]', y=nEvts, rebin=10),
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
        vars1D.update({
            # PFMET-specific
            'HFEMEt'                  : HLTMETInputs(x=r'HF EM $E_T$  [GeV]', y=nEvts),
            'HFEMEtFraction'          : HLTMETInputs(x=r'HF EM $E_T$ fraction', y=nEvts),
            'HFHadronEt'              : HLTMETInputs(x=r'HF Hadronic $E_T$ [GeV]', y=nEvts, rebin=2),
            'HFHadronEtFraction'      : HLTMETInputs(x='HF Hadronic $E_T$ fraction', y=nEvts, rebin=2),
            'chargedHadronEt'         : HLTMETInputs(x=r'Charged Hadron $E_T$ [GeV]', y=nEvts, rebin=2),
            'chargedHadronEtFraction' : HLTMETInputs(x=r'Charged Hadron $E_T$ fraction', y=nEvts, rebin=2),
            'neutralHadronEt'         : HLTMETInputs(x=r'Neutral Hadron $E_T$ [GeV]', y=nEvts, rebin=2),
            'neutralHadronEtFraction' : HLTMETInputs(x=r'Neutral Hadron $E_T$ fraction', y=nEvts, rebin=2),
            'photonEt'                : HLTMETInputs(x=r'Photon $E_T$ [GeV]', y=nEvts, rebin=2),
            'photonEtFraction'        : HLTMETInputs(x=r'Photon $E_T$ fraction', y=nEvts, rebin=2),
            'muonEt'                  : HLTMETInputs(x=r'Muon $E_T$ [GeV]', y=nEvts),
            'muonEtFraction'          : HLTMETInputs(x=r'Muon $E_T$ fraction', y=nEvts),
            'electronEt'              : HLTMETInputs(x=r'Electron $E_T$ [GeV]', y=nEvts),
            'electronEtFraction'      : HLTMETInputs(x=r'Electron $E_T$ fraction', y=nEvts),
        })
    if isMHT:
        for k in vars1D:
            vars1D[k].x = vars1D[k].x.replace('MET', 'MHT')
            vars1D[k].y = vars1D[k].y.replace('MET', 'MHT')

    vars2D = {
        'GenMETTruevsMET': HLTMETInputs(x=r'$MET_\text{Gen}$ [GeV]', y=r'$MET_\text{Reco}$ [GeV]', drawdiag=True),
        'GenMETPhivsMET': HLTMETInputs(x=r'$\text{MET}\phi_\text{Gen}$', y=r'$MET_\text{Reco}$ [GeV]'),
        'GenMETTruevsGenMETPhi': HLTMETInputs(x=r'$MET_\text{Gen}$ [GeV]', y=r'$\text{MET}\phi_\text{Gen}$'),
        'METDiffvsGenMETTrue': HLTMETInputs(x=r'$\text{MET}_\text{Reco} - \text{MET}_\text{Gen}$ [GeV]', y=r'$\text{MET}_\text{Gen}$ [GeV]'),
        'METDiffvsGenMETPhi': HLTMETInputs(x=r'$\text{MET}_\text{Reco} - \text{MET}_\text{Gen}$ [GeV]', y=r'$\text{MET}\phi_\text{Gen}$'),
        'METRatiovsGenMETTrue': HLTMETInputs(x=r'$\text{MET}_\text{Reco} / \text{MET}_\text{Gen}$', y=r'$\text{MET}_\text{Gen}$ [GeV]'),
        'METRatiovsGenMETPhi': HLTMETInputs(x=r'$\text{MET}_\text{Reco} / \text{MET}_\text{Gen}$', y=r'$\text{MET}\phi_\text{Gen}$'),
        'METDeltaPhivsGenMETTrue': HLTMETInputs(x=r'$\text{MET}\phi_\text{Reco} - \text{MET}\phi_\text{Gen}$', y=r'$\text{MET}_\text{Gen}$ [GeV]'),
        'METDeltaPhivsGenMETPhi': HLTMETInputs(x=r'$\text{MET}\phi_\text{Reco} - \text{MET}\phi_\text{Gen}$', y=r'$\text{MET}\phi_\text{Gen}$'),
    }

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
    }
    
    colors = hep.style.CMS['axes.prop_cycle'].by_key()['color']
    markers = ('o', 's', 'd')
    errorbar_kwargs = dict(capsize=3, elinewidth=0.8, capthick=2, linewidth=2, linestyle='')

    nEvts = '# Events'

    if args.compare_collections is not None:
        afile = ROOT.TFile.Open(args.file)
        dqm_dir = f"DQMData/Run 1/HLT/Run summary/JetMET/METValidation"
        checkRootDir(afile, dqm_dir)
        plot1DCollectionComparison(afile, dqm_dir, varsToPlot(args.compare_collections[0])[0],
                                   outdir=args.odir, metTypes=args.compare_collections)

    elif args.compare_files is not None:
        dqm_dir = f"DQMData/Run 1/HLT/Run summary/JetMET/METValidation"
        for afile, alabel in zip(args.compare_files, args.compare_files_labels):
            afile = ROOT.TFile.Open(afile)
            checkRootDir(afile, dqm_dir)
        plot1DFilesComparison(dqm_dir, varsToPlot(args.met[0])[0], outdir=args.odir, metType=args.met[0],
                              files=args.compare_files,
                              files_labels=args.compare_files_labels)

    else:
        afile = ROOT.TFile.Open(args.file)
        dqm_dir = f"DQMData/Run 1/HLT/Run summary/JetMET/METValidation"

        # Plot 1D MET variables
        for metType in args.met:
            dqm_dir_met = os.path.join(dqm_dir, metType)
            checkRootDir(afile, dqm_dir_met)
            plotVars(afile, dqm_dir_met, varsToPlot(metType)[0], outdir=os.path.join(args.odir, metType),
                     mode='1D', metType=METType[metType])
            plotVars(afile, dqm_dir_met, varsToPlot(metType)[1], outdir=os.path.join(args.odir, metType),
                     mode='2D', metType=METType[metType])

        # Plot MET vs MHT comparisons, one folder per MET-MHT pair
        metvsmht_dir = afile.Get(os.path.join(dqm_dir, 'METvsMHT'))
        if metvsmht_dir:
            for key in metvsmht_dir.GetListOfKeys():
                pair = key.GetName()
                plotVars(afile, os.path.join(dqm_dir, 'METvsMHT', pair), varsToPlotMETvsMHT(), mode='2D',
                         outdir=os.path.join(args.odir, 'METvsMHT', pair), metType=pair.replace('_', ' vs '))
