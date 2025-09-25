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

def checkRootFile(hname, rebin=None):
    hist_orig = file.Get(hname)
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
                
    def save(self, name):
        for ext in self.extensions:
            print(" ### INFO: Saving " + name + '.' + ext)
            plt.savefig(name + '.' + ext)
        plt.close()

def plot1Dvars(adir, avars, outdir, metType, top_text=False):
    """
    Plots 1D distributions.
    The `avars` variables is a dictionary whose values are (xlabel, ylabel, rebin).
    """
    for var, (xlabel, ylabel, rebin) in avars.items():
        plotter = Plotter(args.sample_label)
        root_hist = checkRootFile(f"{adir}/{var}", rebin=rebin)
        nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
        values, errors = histo_values_errors(root_hist)

        plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                            fmt='s', color='black', label=xlabel, **errorbar_kwargs)
        # plotter.ax.step(bin_edges[:-1], values, where="post", color='black')
        plotter.ax.stairs(values, bin_edges, color='black', linewidth=2, baseline=None)
        plotter.ax.text(0.03, 0.97, metType, transform=plotter.ax.transAxes, fontsize=fontsize,
                        verticalalignment='top', horizontalalignment='left')

        diff_step = 0.05 * abs(max(values)-min(values))
        plotter.limits(y=(min(values) - diff_step, max(values) + 2*diff_step), logY=False)
        plotter.labels(x=xlabel, y=ylabel)

        if top_text:
            plotter.ax.text(0.97, 0.97, root_hist.GetTitle().replace('ET', r'$E_T$'), transform=plotter.ax.transAxes, fontsize=fontsize,
                            verticalalignment='top', horizontalalignment='right')

        plotter.save( os.path.join(outdir, var) )

def plot1Dtrigger(adir, avars, metType, outdir):
    avarsDict = {}
    for var in avars:
        root_hist = checkRootFile(f"{adir}/{var}", rebin=None)

        xlabel = root_hist.GetXaxis().GetTitle().replace('ET', r'$E_T$ ')
        ylabel = root_hist.GetYaxis().GetTitle()

        avarsDict[var] = (xlabel, ylabel, None)

    plot1Dvars(adir, avarsDict, outdir, metType=metType, top_text=True)


def plot1Dcomparison(adir, avars, outdir, metTypes):
    """
    Plots 1D distributions.
    The `avars` variables is a dictionary whose values are (xlabel, ylabel, rebin).
    """
    outdir = os.path.join(outdir, 'Comparison_' + '_'.join(metTypes))
    createDir(outdir)

    for var, (xlabel, ylabel, rebin) in avars.items():
        plotter = Plotter(args.sample_label)

        for metType in metTypes:
            root_hist = checkRootFile(f"{adir}/{metType}/{var}", rebin=rebin)
            nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
            values, errors = histo_values_errors(root_hist)

            patch = plotter.ax.stairs(values, bin_edges, linewidth=2, baseline=None)
            plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                                fmt='s', label=metType, color=patch.get_edgecolor(),
                                **errorbar_kwargs)

        plotter.ax.legend()
        
        diff_step = 0.05 * abs(max(values)-min(values))
        plotter.limits(y=(min(values) - diff_step, max(values) + 2*diff_step), logY=False)
        plotter.labels(x=xlabel, y=ylabel)

        plotter.save( os.path.join(outdir, var) )

if __name__ == '__main__':

    def check_list_length(value):
        if len(value) <= 1:
            raise argparse.ArgumentTypeError("List must have more than one item")
        return value

    full_command = 'for amet in "hltPFMET" "hltPFPuppiMET" "hltPFPuppiMETTypeOne"; do python3 Validation/RecoMET/scripts/makeHLTMETValidationPlots.py --file Run/DQM_1000_Wprime.root --odir /eos/user/b/bfontana/www/MET_Valid/Wprime -l Wprime --met ${amet}; done'
    parser = argparse.ArgumentParser(description='Make HLT MET validation plots. \nRun all MET paths with\n' + full_command)
    parser.add_argument('-f', '--file', required=True, help='Paths to the DQM ROOT file.')
    parser.add_argument('-o', '--odir', default="HLTMETValidationPlots", required=False, help='Path to the output directory.')
    parser.add_argument('-l', '--sample_label', default="QCD (200 PU)", required=False,  help='Sample label for plotting.')

    mutual_excl = parser.add_mutually_exclusive_group(required=True)
    mutual_excl.add_argument('-m', '--met',  nargs='+', default='hltPFPuppiMET', required=False, help='Name of the met collection(s).')
    mutual_excl.add_argument('-c', '--comparison',  nargs='+', default=None,
                             choices=('hltPFMET', 'hltPFPuppiMET', 'hltPFPuppiMETTypeOne'),
                             type=check_list_length, required=False,
                             help='Name of the met collection(s).', )
    
    args = parser.parse_args()

    createDir(args.odir)
    for metType in args.met:
        outdir = createDir(os.path.join(args.odir, metType))
    
    file = ROOT.TFile.Open(args.file)
    fontsize = 16
    tprofile_rebinning = {'B': (30, 40, 50, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600), #barrel
                          'E': (30, 40, 50, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600), # endcap
                          'F': (30, 40, 50, 80, 120, 240, 600)} # forward

    METType = {'hltPFMET': "PF MET",
               'hltPFPuppiMET': "PF PUPPI MET",
               'hltPFPuppiMETTypeOne': "PF Type-1 PUPPI MET"}
    
    colors = hep.style.CMS['axes.prop_cycle'].by_key()['color']
    markers = ('o', 's', 'd')
    errorbar_kwargs = dict(capsize=3, elinewidth=0.8, capthick=2, linewidth=2, linestyle='')

    nEventsLabel = '# Events'
    vars1D = {
        # MET tester producer
        'HFEMEt'                  : (r'HF EM $E_T$', nEventsLabel, None),
        'HFEMEtFraction'          : (r'HF EM $E_T$ fraction', nEventsLabel, None),
        'HFHadronEt'              : (r'HF Hadron $E_T$', nEventsLabel, 2),
        'HFHadronEtFraction'      : ('HF Hadron $E_T$ fraction', nEventsLabel, 2),
        'MET'                     : ('MET', nEventsLabel, 2),
        'MEx'                     : ('MET x', nEventsLabel, 4),
        'MEy'                     : ('MET y', nEventsLabel, 4),
        'METPhi'                  : (r'MET $\phi$', nEventsLabel, 2),
        'METDeltaPhi_GenMETCalo'  : (r'Calo MET $\Delta\phi$', nEventsLabel, 2),
        'METDeltaPhi_GenMETTrue'  : (r'True MET $\Delta\phi$', nEventsLabel, 2),
        'METDiff_GenMETCalo'      : (r'MET - gen MET$_{Calo}$', nEventsLabel, 10),
        'METDiff_GenMETTrue'      : (r'MET - gen MET$_{True}$', nEventsLabel, 10),
        'METSignPseudo'           : ('MET Significance (Event-by-event)', nEventsLabel, None), # Et / std: (: (sqrt(sumEt)
        'METSignReal'             : ('MET Significance (Likelihood)', nEventsLabel, None), # covariance matrix missing
        'MET_Nvtx'                : ('Number of vertices (MET-weighted)', nEventsLabel, 6),
        'Nvertex'                 : ('Number of vertices', nEventsLabel, 6),
        'SumET'                   : (r'$\sum E_T$', nEventsLabel, 4),
        'chargedHadronEt'         : (r'Charged Hadron $E_T$', nEventsLabel, 2),
        'chargedHadronEtFraction' : (r'Charged Hadron $E_T$ fraction', nEventsLabel, 2),
        'neutralHadronEt'         : (r'Neutral Hadron $E_T$', nEventsLabel, 2),
        'neutralHadronEtFraction' : (r'Neutral Hadron $E_T$ fraction', nEventsLabel, 2),
        'photonEt'                : (r'Photon $E_T$', nEventsLabel, 2),
        'photonEtFraction'        : (r'Photon $E_T$ fraction', nEventsLabel, 2),
        'muonEt'                  : (r'Muon $E_T$', nEventsLabel, None),
        'muonEtFraction'          : (r'Muon $E_T$ fraction', nEventsLabel, None),
        'electronEt'              : (r'Electron $E_T$', nEventsLabel, None),
        'electronEtFraction'      : (r'Electron $E_T$ fraction', nEventsLabel, None),
        # MET post-processing
        'METDiffAggr_MET': ('MET', 'MET Mean Difference', None),
        'METDiffAggr_Phi': (r'$\phi$', 'MET Mean Difference', None),
        'METResolAggr_MET': ('MET', 'MET Resolution', None),
        'METResolAggr_Phi': (r'$\phi$', 'MET Resolution', None),
        'METRespAggr_MET': ('MET', 'MET Response', None),
        'METRespAggr_Phi': (r'$\phi$', 'MET Response', None),
        'METSignAggr_MET': ('MET', 'MET Significance', None),
        'METSignAggr_Phi': (r'$\phi$', 'MET Significance', None)
    }

    
    dqm_dir = f"DQMData/Run 1/HLT/Run summary/JetMET/METValidation"
    if args.comparison is None:
        # Plot 1D MET variables
        for metType in args.met:
            dqm_dir_met = os.path.join(dqm_dir, metType)
            checkRootDir(file, dqm_dir_met)
            plot1Dvars(dqm_dir_met, vars1D, outdir=os.path.join(args.odir, metType),
                       metType=METType[metType])

        # Plot MET turn-on curves
        trigger = 'HLT_PFPuppiMETTypeOne140_PFPuppiMHT140'
        turnon_dir = f"DQMData/Run 1/HLT/Run summary/JetMET/TurnOnValidation/{trigger}"
        checkRootDir(file, turnon_dir)
        vars1Dtrigger = ('TurnOngMET', 'TurnOngMETLow', 'TurnOnhMET', 'TurnOnhMETLow')
        outdir = createDir(os.path.join(args.odir, trigger))
        plot1Dtrigger(turnon_dir, vars1Dtrigger, outdir=outdir, metType=METType[metType])
    else:
        checkRootDir(file, dqm_dir)
        plot1Dcomparison(dqm_dir, vars1D, outdir=args.odir, metTypes=args.comparison)
        
