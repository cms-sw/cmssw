#!/usr/bin/env python3

import os
import argparse
import numpy as np
import hist
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
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
    
def CheckRootFile(hname, rebin=None):
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

if __name__ == '__main__':
    full_command = 'for amet in "hltPFMET" "hltPFPuppiMET" "hltPFPuppiMETTypeOne"; do python3 Validation/RecoMET/scripts/makeHLTMETValidationPlots.py --file Run/DQM_1000_Wprime.root --odir /eos/user/b/bfontana/www/MET_Valid/Wprime -l Wprime --met ${amet}; done'
    parser = argparse.ArgumentParser(description='Make HLT MET validation plots. \nRun all MET paths with\n' + full_command)
    parser.add_argument('-f', '--file', required=True, help='Paths to the DQM ROOT file.')
    parser.add_argument('-m', '--met', default='hltPFPuppiMET', help='Name of the met collection')
    parser.add_argument('-o', '--odir', default="HLTMETValidationPlots", required=False, help='Path to the output directory.')
    parser.add_argument('-l', '--sample_label', default="QCD (200 PU)", required=False,  help='Sample label for plotting.')
    args = parser.parse_args()

    if not os.path.exists(args.odir):
        os.makedirs(args.odir)

    outdir = os.path.join(args.odir, args.met)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    file = ROOT.TFile.Open(args.file)
    dqm_dir = f"DQMData/Run 1/HLT/Run summary/JetMET/METValidation/{args.met}"
    if not file.Get(dqm_dir):
        raise RuntimeError(f"Directory '{dqm_dir}' not found in {args.file}")

    fontsize = 16    

    tprofile_rebinning = {'B': (30, 40, 50, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600), #barrel
                          'E': (30, 40, 50, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600), # endcap
                          'F': (30, 40, 50, 80, 120, 240, 600)} # forward

    METType = {'hltPFMET': "PF MET",
               'hltPFPuppiMET': "PF PUPPI MET",
               'hltPFPuppiMETTypeOne': "PF Type-1 PUPPI MET"}.get(args.met, args.met)
    
    colors = hep.style.CMS['axes.prop_cycle'].by_key()['color']
    markers = ('o', 's', 'd')
    errorbar_kwargs = dict(capsize=3, elinewidth=0.8, capthick=2, linewidth=2, linestyle='')

    #####################################
    # Plot 1D variables from METTester
    #####################################
    Var1DList = {
        'HFEMEt'                  : (r'HF EM $E_T$', None),
        'HFEMEtFraction'          : (r'HF EM $E_T$ fraction', None),
        'HFHadronEt'              : (r'HF Hadron $E_T$', 2),
        'HFHadronEtFraction'      : ('HF Hadron $E_T$ fraction', 2),
        'MET'                     : ('MET', 2),
        'MEx'                     : ('MET x', 4),
        'MEy'                     : ('MET y', 4),
        'METPhi'                  : (r'MET $\phi$', 2),
        'METDeltaPhi_GenMETCalo'  : (r'Calo MET $\Delta\phi$', 2),
        'METDeltaPhi_GenMETTrue'  : (r'True MET $\Delta\phi$', 2),
        'METDiff_GenMETCalo'      : (r'MET - gen MET$_{Calo}$', 10),
        'METDiff_GenMETTrue'      : (r'MET - gen MET$_{True}$', 10),
        'METSignPseudo'           : ('MET Significance (Event-by-event)', None), # Et / std: (: (sqrt(sumEt)
        'METSignReal'             : ('MET Significance (Likelihood)', None), # covariance matrix missing
        'MET_Nvtx'                : ('Number of vertices (MET-weighted)', 6),
        'Nvertex'                 : ('Number of vertices', 6),
        'SumET'                   : (r'$\sum E_T$', 4),
        'chargedHadronEt'         : (r'Charged Hadron $E_T$', 2),
        'chargedHadronEtFraction' : (r'Charged Hadron $E_T$ fraction', 2),
        'neutralHadronEt'         : (r'Neutral Hadron $E_T$', 2),
        'neutralHadronEtFraction' : (r'Neutral Hadron $E_T$ fraction', 2),
        'photonEt'                : (r'Photon $E_T$', 2),
        'photonEtFraction'        : (r'Photon $E_T$ fraction', 2),
        'muonEt'                  : (r'Muon $E_T$', None),
        'muonEtFraction'          : (r'Muon $E_T$ fraction', None),
        'electronEt'              : (r'Electron $E_T$', None),
        'electronEtFraction'      : (r'Electron $E_T$ fraction', None),
    }
    
    for var, (xlabel, rebin) in Var1DList.items():
        plotter = Plotter(args.sample_label)
        root_hist = CheckRootFile(f"{dqm_dir}/{var}", rebin=rebin)
        nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
        values, errors = histo_values_errors(root_hist)

        plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                     fmt='s', color='black', label=xlabel, **errorbar_kwargs)
        plotter.ax.step(bin_edges[:-1], values, where="post", color='black')
        plotter.ax.text(0.03, 0.97, METType, transform=plotter.ax.transAxes, fontsize=fontsize,
                        verticalalignment='top', horizontalalignment='left')

        diff_step = 0.05 * abs(max(values)-min(values))
        plotter.limits(y=(min(values) - diff_step, max(values) + 2*diff_step), logY=False)
        plotter.labels(x=xlabel, y='# Events')

        plotter.save( os.path.join(outdir, var) )

    ################################################
    # Plot 1D variables from METTesterPostProcessor
    ################################################
    var1dNames = {'METDiffAggr_MET': ('MET', 'MET Mean Difference'),
                  'METDiffAggr_Phi': (r'$\phi$', 'MET Mean Difference'),
                  'METResolAggr_MET': ('MET', 'MET Resolution'),
                  'METResolAggr_Phi': (r'$\phi$', 'MET Resolution'),
                  'METRespAggr_MET': ('MET', 'MET Response'),
                  'METRespAggr_Phi': (r'$\phi$', 'MET Response'),
                  'METSignAggr_MET': ('MET', 'MET Significance'),
                  'METSignAggr_Phi': (r'$\phi$', 'MET Significance')}

    for var, (xlabel, ylabel) in var1dNames.items():
        plotter = Plotter(args.sample_label)
        root_hist = CheckRootFile(f"{dqm_dir}/{var}", rebin=None)
        nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
        values, errors = histo_values_errors(root_hist)
        
        plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                             fmt='s', color='black', label=xlabel, **errorbar_kwargs)
        plotter.ax.stairs(values, bin_edges, color='black', linewidth=2)
        plotter.ax.text(0.03, 0.97, METType, transform=plotter.ax.transAxes, fontsize=fontsize,
                        verticalalignment='top', horizontalalignment='left')

        diff_step = 0.05 * abs(max(values)-min(values))
        plotter.limits(y=(min(values) - diff_step, max(values) + 2*diff_step), logY=False)
        plotter.labels(x=xlabel, y=ylabel)

        plotter.save( os.path.join(outdir, var) )

    ######################
    # Plot turn-on curves
    ######################
    turnon_dir = f"DQMData/Run 1/HLT/Run summary/JetMET/TurnOnValidation/HLT_PFPuppiMETTypeOne140_PFPuppiMHT140"
    if not file.Get(dqm_dir):
        raise RuntimeError(f"Directory '{turnon_dir}' not found in {args.file}")

    var1dNames = ('TurnOngMET', 'TurnOngMETLow', 'TurnOnhMET', 'TurnOnhMETLow')

    for var in var1dNames:
        plotter = Plotter(args.sample_label)
        root_hist = CheckRootFile(f"{turnon_dir}/{var}", rebin=None)
        nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
        values, errors = histo_values_errors(root_hist)

        xlabel = root_hist.GetXaxis().GetTitle().replace('ET', r'$E_T$ ')
        ylabel = root_hist.GetYaxis().GetTitle()
        title = root_hist.GetTitle()

        plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                             fmt='s', color='black', label=xlabel, **errorbar_kwargs)
        plotter.ax.stairs(values, bin_edges, color='black', linewidth=2)
        plotter.ax.text(0.03, 0.97, METType, transform=plotter.ax.transAxes, fontsize=fontsize,
                        verticalalignment='top', horizontalalignment='left')

        diff_step = 0.05 * (max(values)-min(values))
        plotter.limits(y=(min(values) - diff_step, 1.2*max(values)), logY=False)
        plotter.labels(x=xlabel, y=ylabel)

        plotter.ax.text(0.97, 0.97, title.replace('ET', r'$E_T$'), transform=plotter.ax.transAxes, fontsize=fontsize,
                        verticalalignment='top', horizontalalignment='right')

        newvar = var.replace('_', '')
        plotter.save( os.path.join(outdir, newvar) )
