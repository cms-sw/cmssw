#!/usr/bin/env python3

import os
import argparse
import numpy as np
import hist
import matplotlib.pyplot as plt
import array
import ROOT
import mplhep as hep
hep.style.use("CMS")

from abc import ABC, abstractmethod

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

def CheckRootFile(hname):
    hist = file.Get(hname)
    if not hist:
        raise RuntimeError(f"WARNING: Histogram {dqm_dir}/{Type}{Var} not found.")
    return hist

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

def histo_values(h, errors=False):
    N = h.GetNbinsX()
    if errors:
        values = np.array([h.GetBinError(i+1) for i in range(N)])
    else:
        values = np.array([h.GetBinContent(i+1) for i in range(N)])
    return values 

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

    def limits(self, x=None, y=None):
        if x:
            self._ax.set_xlim(x)
        if y:
            self._ax.set_ylim(y)
                
    def save(self, name):
        for ext in self.extensions:
            print(" ### INFO: Saving " + name + '.' + ext)
            plt.savefig(name + '.' + ext)
        plt.close()
    
class EtaInfo:
    """
    Manage information related with the eta region.
    """
    info = {
        'B': ('black', 'o', r'$|\eta|<1.5$'),
        'E': ('red',   's', r'$1.5\leq|\eta|<3$'),
        'F': ('blue',  's', r'$3\leq|\eta|<6$')
    }

    @staticmethod
    def color(x):
        return EtaInfo.info.get(x)[0]
    
    @staticmethod
    def marker(x):
        return EtaInfo.info.get(x)[1]

    @staticmethod
    def label(x):
        return EtaInfo.info.get(x)[2]

    @staticmethod
    def regions():
        return ('B', 'E', 'F')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make HLT Jet validation plots.')
    parser.add_argument('-f', '--file', type=str, required=True,                                   help='Paths to the DQM ROOT file.')
    parser.add_argument('-j', '--jet',  type=str, default='hltAK4PFPuppiJets',                     help='Name of the jet collection')
    parser.add_argument('-o', '--odir', type=str, default="HLTJetValidationPlots", required=False, help='Path to the output directory.')
    parser.add_argument('-l', '--sample_label', type=str, default="QCD (200 PU)", required=False,  help='Path to the output directory.')
    args = parser.parse_args()

    if not os.path.exists(args.odir):
        os.makedirs(args.odir)

    file = ROOT.TFile.Open(args.file)
    dqm_dir = f"DQMData/Run 1/HLT/Run summary/JetMET/JetValidation/{args.jet}"
    if not file.Get(dqm_dir):
        raise RuntimeError(f"Directory '{dqm_dir}' not found in {args.file}")

    fontsize = 16       

    ResLabels = ('PtRecoOverGen', 'PtCorrOverGen', 'PtCorrOverReco')
    PtLabels = ("20 < $p_T$ < 40 GeV", "40 < $p_T$ < 100 GeV", "100 < $p_T$ < 300 GeV", "300 < $p_T$ < 600 GeV", "600 < $p_T$ < 5000 GeV")

    tprofile_rebinning = {'B': (30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600), #barrel
                          'E': (30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600), # endcap
                          'F': (30, 40, 50, 60, 80, 120, 240, 600)} # forward

    if args.jet == 'hltAK4PFPuppiJets': JetType = "AK4 PF Puppi Jets"
    elif args.jet == 'hltAK4PFClusterJets': JetType = "AK4 PF Cluster Jets"
    elif args.jet == 'hltAK4PFJets': JetType = "AK4 PF Jets"
    elif args.jet == 'hltAK4PFCHSJets': JetType = "AK4 PF CHS Jets"
    else: JetType = args.jet

    #####################################
    # Response
    #####################################

    for i_res, ResType in enumerate(('PtRecoOverGen', 'PtCorrOverGen', 'PtCorrOverReco')):

        for EtaRegion in EtaInfo.regions():

            PtBinsToMerge = [
                [f'h_{ResType}_{EtaRegion}_Pt20_30', f'h_{ResType}_{EtaRegion}_Pt30_40'],
                [f'h_{ResType}_{EtaRegion}_Pt40_100',],
                [f'h_{ResType}_{EtaRegion}_Pt100_200', f'h_{ResType}_{EtaRegion}_Pt200_300'],
                [f'h_{ResType}_{EtaRegion}_Pt300_600',],
                [f'h_{ResType}_{EtaRegion}_Pt600_2000', f'h_{ResType}_{EtaRegion}_Pt2000_5000']
            ]

            v_stacked_histo = []
            for i, hist_names in enumerate(PtBinsToMerge):
                axis = None
                stacked = None

                for j, hist_name in enumerate(hist_names):                    
                    root_hist = CheckRootFile(f"{dqm_dir}/{hist_name}")
                    
                    # root_hist = root_hist.Rebin(2) # [FIXME] due to low statistics
                    nbins = root_hist.GetNbinsX()
                    edges = [root_hist.GetBinLowEdge(i+1) for i in range(nbins)]
                    edges.append(root_hist.GetBinLowEdge(nbins+1))
                    values = np.array([root_hist.GetBinContent(i+1) for i in range(nbins)])
                    errors = np.array([root_hist.GetBinError(i+1) for i in range(nbins)])
                    label = root_hist.GetXaxis().GetTitle()

                    if axis is None:
                        axis = hist.axis.Variable(edges, name=root_hist.GetTitle(), label=label)

                    h = hist.Hist(axis, storage=hist.storage.Weight())
                    h.view().value[:] = values
                    h.view().variance[:] = errors**2

                    if stacked is None:
                        stacked = h
                    else:
                        stacked += h

                if stacked == None: continue

                v_stacked_histo.append(stacked)

            if len(v_stacked_histo) == 0: continue

            plotter = Plotter(args.sample_label)

            for i, stacked_histo in enumerate(v_stacked_histo):
                if stacked_histo.sum().value == 0:
                    print(f"WARNING: Skipping empty histogram for {PtLabels[i]}")
                    continue
                stacked_histo.plot(ax=plotter.ax, linewidth=2, label=PtLabels[i])
                
            plotter.ax.text(0.03, 0.97, f"{JetType}\n{EtaInfo.label(EtaRegion)}", transform=plotter.ax.transAxes, fontsize=fontsize,
                            verticalalignment='top', horizontalalignment='left')

            plotter.labels(x="${}$".format(v_stacked_histo[0].axes[0].label),
                           y="# Jets",
                           legend_title=r"Jet $p_T$ range ")

            plotter.save( os.path.join(args.odir, f"Response{ResType}_{EtaRegion}") )

    
    #####################################
    # Response vs pt from profile
    #####################################

    x_axis_titles = ("$p_{T}^{gen}\,$", "$p_{T}^{gen}\,$", "$p_{T}^{reco}\,$")
    y_axis_titles = ("$p_{T}^{reco}/p_{T}^{gen}\,$", "$p_{T}^{corr}/p_{T}^{gen}\,$", "$p_{T}^{corr}/p_{T}^{reco}\,$")
    for i_res, ResType in enumerate(('PtRecoOverGen_GenPt', 'PtCorrOverGen_GenPt', 'PtCorrOverReco_Pt')):

        plotter = Plotter(args.sample_label)

        for EtaRegion in EtaInfo.regions():
            tprofile = CheckRootFile(f"{dqm_dir}/pr_{ResType}_{EtaRegion}")

            bin_edges_c = array.array('d', tprofile_rebinning[EtaRegion])
            tprofile = tprofile.Rebin(len(tprofile_rebinning[EtaRegion]) - 1, "tprofile", bin_edges_c)

            nbins, bin_edges, bin_centers, bin_widths = define_bins(tprofile)

            means = histo_values(tprofile)
            mean_errors = histo_values(tprofile, errors=True)

            plotter.ax.errorbar(bin_centers, means, xerr=0.5 * bin_widths, yerr=mean_errors, linestyle='',
                                fmt=EtaInfo.marker(EtaRegion), color=EtaInfo.color(EtaRegion), label=EtaInfo.label(EtaRegion))
            plotter.ax.step(bin_edges[:-1], means, where="post", color=EtaInfo.color(EtaRegion), linestyle='-', linewidth=2)

        plotter.ax.text(0.03, 0.97, f"{JetType}", transform=plotter.ax.transAxes, fontsize=fontsize,
                        verticalalignment='top', horizontalalignment='left')
        plotter.labels(x=x_axis_titles[i_res]+" [GeV]", y="<{}>".format(y_axis_titles[i_res]),
                       legend_title='')
        plotter.limits(y=(0.0, 2.7))

        plotter.save( os.path.join(args.odir, 'Response_' + ResType) )
    
    #####################################
    # Resolution vs pt from profile
    #####################################

    names = ('pr_PtCorrOverGen_GenEta', 'pr_PtCorrOverGen_GenPt_B', 'pr_PtCorrOverGen_GenPt_E', 'pr_PtCorrOverGen_GenPt_F')

    x_axis_titles = ("$p_{T}^{gen}$", "$p_{T}^{gen}$", "$p_{T}^{reco}$")
    y_axis_titles = ("$p_{T}^{reco}/p_{T}^{gen}$", "$p_{T}^{corr}/p_{T}^{gen}$", "$p_{T}^{corr}/p_{T}^{reco}$")
    for i_res, ResType in enumerate(['PtRecoOverGen_GenPt', 'PtCorrOverGen_GenPt', 'PtCorrOverReco_Pt']):

        plotter = Plotter(args.sample_label)
        
        for EtaRegion in EtaInfo.regions():
            tprofile = CheckRootFile(f"{dqm_dir}/pr_{ResType}_{EtaRegion}")

            bin_edges_c = array.array('d', tprofile_rebinning[EtaRegion][::2]) # [FIXME] due to low statistics
            tprofile = tprofile.Rebin(len(tprofile_rebinning[EtaRegion][::2]) - 1, "tprofile", bin_edges_c)
            nbins, bin_edges, bin_centers, bin_widths = define_bins(tprofile)

            means = histo_values(tprofile)
            sigmas = np.array([tprofile.GetBinError(i+1) * np.sqrt(tprofile.GetBinEntries(i+1)) for i in range(nbins)])
            sigma_errors = np.array([tprofile.GetBinError(i+1) * np.sqrt(2) for i in range(nbins)])
            resolution = [s / m if m != 0 else np.nan for s, m in zip(sigmas, means)]

            plt.errorbar(bin_centers, resolution, xerr=0.5 * bin_widths, yerr=sigma_errors, linestyle='',
                         fmt=EtaInfo.marker(EtaRegion), color=EtaInfo.color(EtaRegion), label=EtaInfo.label(EtaRegion))
            plt.step(bin_edges[:-1], resolution, where="post", color=EtaInfo.color(EtaRegion), linestyle='-', linewidth=2)

        plotter._ax.text(0.03, 0.97, f"{JetType}", transform=plotter._ax.transAxes, fontsize=fontsize,
                 verticalalignment='top', horizontalalignment='left')

        plotter.labels(x=x_axis_titles[i_res]+" [GeV]",
                       y=f"$\sigma$({y_axis_titles[i_res]}) / <{y_axis_titles[i_res]}>",
                       legend_title='')

        plotter.limits(y=(0,0.8))
        plotter.save( os.path.join(args.odir, 'Resolution_' + ResType) )

    ########################################
    # Jet efficiency, fakes and duplicates
    ########################################

    den_label = ('HLT Jets $p_T > 30$ GeV', 'Gen Jets $p_T > 20$ GeV')
    num_label = ('HLT Jets $p_T > 30$ GeV matched to gen jets', 'Gen Jets $p_T > 20$ GeV matched to HLT jets')
    y_title = ('Jet Mistag Rate', )
    eff_color = '#bd1f01'

    class HLabels:
        def __init__(self, atype):
            types = ('Efficiency', 'Fake Rate', 'Gen Duplicates', 'Reco Duplicates')
            assert atype in types
            self.mytype = atype

        @staticmethod
        def types():
            return ('Efficiency', 'Fake Rate', 'Gen Duplicates', 'Reco Duplicates')
        
        @property
        def ytitle(self):
            if self.mytype == 'Efficiency':
                return 'Jet Efficiency'
            elif self.mytype == 'Fake Rate':
                return 'Jet Fake Rate'
            elif self.mytype == 'Jet Gen Duplicates':
                return 'Jet Gen Duplicates'
            elif self.mytype == 'Jet Reco Duplicates':
                return 'Jet Reco Duplicates'

        @property
        def xvars(self):
            if 'Duplicates' in self.mytype:
                return ('Pt_B', 'Pt_E', 'Pt_F',
                        'Phi_B', 'Phi_E', 'Phi_F',
                        'Eta', 'Phi', 'Pt')
            else:
                return ('Pt_B', 'Pt_E', 'Pt_F',
                        'Eta', 'Phi', 'Pt'):

        def hname(self, var, basename, isNum):
            name = basename
            if self.mytype == 'Efficiency':
                name += 'JetEfficiency' + var if isNum else ""
            elif self.mytype == 'Fake Rate':
                name += 'JetMistagRate' + var
            elif self.mytype == 'Gen Duplicates':
                name += 'JetMistagRate' + var

    for Type in HLabels.types:
        hlabel = HLabel(Type)
        for Var in hlabel.xvars:
            root_hist = CheckRootFile( hlabel.hname(Var, dqm_dir, False) )
            root_hist_matched = CheckRootFile( hlabel.hname(Var, dqm_dir, True) )

            # root_hist = CheckRootFile(f"{dqm_dir}/{Type}{Var}")
            # root_hist_matched = CheckRootFile(f"{dqm_dir}/Matched{Type}{Var}")

            h_ratio = root_hist.Clone("h_ratio")
            h_ratio.Divide(root_hist_matched, root_hist, 1.0, 1.0, "B")

            nbins, bin_edges, bin_centers, bin_widths = define_bins(h_ratio)
            numerator_vals = histo_values(root_hist_matched)
            denominator_vals = histo_values(root_hist)

            if Type == "fake": 
                eff_values = np.array([1 - h_ratio.GetBinContent(i+1) if h_ratio.GetBinContent(i+1) != 0 else 0 for i in range(nbins)])
            elif Type == "eff":
                eff_values = np.array([h_ratio.GetBinContent(i+1) for i in range(nbins)])
            eff_errors = np.array([h_ratio.GetBinError(i+1) for i in range(nbins)])
            label = root_hist.GetXaxis().GetTitle().replace('#', '\\')

            plotter = Plotter(args.sample_label, grid_color=None)
            
            plotter.ax.step(bin_edges[:-1], denominator_vals, where="post", label=den_label[i_type], linewidth=2, color="black")
            plotter.ax.step(bin_edges[:-1], numerator_vals, where="post", label=num_label[i_type], linewidth=2, color="#9c9ca1", linestyle='-.')
            plotter.ax.fill_between(bin_edges[:-1], numerator_vals, step="post", alpha=0.3, color="#9c9ca1")

            plotter.labels(x=f"${label}$", y="# Jets", legend_title='', legend_loc='upper left')
            plotter.limits(y=(0, 1.2*max(denominator_vals)))

            if 'Pt' in Var:
                plotter.ax.text(0.97, 0.97, f"{EtaInfo.label(Var[-1])}", transform=plotter.ax.transAxes, fontsize=fontsize,
                                verticalalignment='top', horizontalalignment='right')

            ax2 = plotter.ax.twinx()
            ax2.errorbar(bin_centers, eff_values, xerr=0.5 * bin_widths, yerr=eff_errors, linestyle='', fmt='o', color=eff_color, label='Efficiency')
            ax2.set_ylabel(hlabels[Type]['y_title'], color=eff_color)
            ax2.set_ylim(0,1.2)
            ax2.grid(color=eff_color, axis='y')
            plotter.ax.grid(color=eff_color, axis='x')

            plotter.save( os.path.join(args.odir,  + '_' + Var) )

        plotter = Plotter(args.sample_label)
        for EtaRegion in EtaInfo.regions():
            root_hist = CheckRootFile(f"{dqm_dir}/{Type}Pt_{EtaRegion}")
            root_hist_matched = CheckRootFile(f"{dqm_dir}/Matched{Type}Pt_{EtaRegion}")

            # root_hist.Rebin(3); root_hist_matched.Rebin(3) # [FIXME] due to very low statistics
            h_ratio = root_hist.Clone("h_ratio")
            h_ratio.Divide(root_hist_matched, root_hist, 1.0, 1.0, "B")

            nbins, bin_edges, bin_centers, bin_widths = define_bins(h_ratio)
            numerator_vals = histo_values(root_hist_matched)
            denominator_vals = histo_values(root_hist)
            
            if i_type == 0: # JetMistagRate = 1 - Matched / Total
                eff_values = np.array([1 - h_ratio.GetBinContent(i+1) if h_ratio.GetBinContent(i+1) != 0 else 0 for i in range(nbins)])
            elif i_type == 1:
                eff_values = histo_values(h_ratio)
            eff_errors = histo_values(h_ratio, errors=True)
            label = root_hist.GetXaxis().GetTitle()

            plt.errorbar(bin_centers, eff_values, xerr=0.5 * bin_widths, yerr=eff_errors, linestyle='',
                         fmt=EtaInfo.marker(EtaRegion), color=EtaInfo.color(EtaRegion), label=EtaInfo.label(EtaRegion))
            plt.step(bin_edges[:-1], eff_values, where="post", color=EtaInfo.color(EtaRegion))

        plotter.labels(x=f"${label}$", y=y_title[i_type], legend_title='')
        plotter.limits(y=(0,1.25))

        plotter.save( os.path.join(args.odir, y_title[i_type].replace(' ', '') + '_Pt') )

    #####################################
    # Plot 1D single variables
    #####################################

    for Var in ('JetPt', 'GenPt', 'CorrJetPt'):
        plotter = Plotter(args.sample_label)
        root_hist = CheckRootFile(f"{dqm_dir}/{Var}")

        nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)

        values = histo_values(root_hist)
        errors = histo_values(root_hist, errors=True)

        plt.errorbar(bin_centers, values, xerr=0.5 * bin_widths, yerr=errors, linestyle='', fmt='s', color='red', linewidth=2)
        plt.step(bin_edges[:-1], values, where="post", color='red')

        plotter.labels(x="$p_T\,$ [GeV]", y=f"# Jets")
        plotter.save( os.path.join(args.odir, Var) )

    #####################################
    # Plot 2D single variables
    #####################################

    Var2DList = ('h2d_PtRecoOverGen_nCost_B', 'h2d_PtRecoOverGen_nCost_E', 'h2d_PtRecoOverGen_nCost_F', 
                 'h2d_PtRecoOverGen_chHad_B', 'h2d_PtRecoOverGen_chHad_E', 'h2d_PtRecoOverGen_chHad_F',
                 'h2d_PtRecoOverGen_neHad_B', 'h2d_PtRecoOverGen_neHad_E', 'h2d_PtRecoOverGen_neHad_F',
                 'h2d_PtRecoOverGen_chEm_B', 'h2d_PtRecoOverGen_chEm_E', 'h2d_PtRecoOverGen_chEm_F',
                 'h2d_PtRecoOverGen_neEm_B', 'h2d_PtRecoOverGen_neEm_E', 'h2d_PtRecoOverGen_neEm_F',)

    for Var2D in Var2DList:
        plotter = Plotter(args.sample_label, fontsize=14)
        root_hist = CheckRootFile(f"{dqm_dir}/{Var2D}")

        nbins_x, nbins_y, x_edges, y_edges = define_bins_2D(root_hist)
        values = histo_values_2D(root_hist)

        x_label = root_hist.GetXaxis().GetTitle().replace('#', '\\')
        y_label = root_hist.GetYaxis().GetTitle().replace('#', '\\')

        # Plot with mplhep's hist2d (preserves ROOT bin edges, color bar included)
        pcm = plotter.ax.pcolormesh(x_edges, y_edges, values, cmap='viridis', shading='auto')

        # Axis labels and style
        plotter.labels(x=f"${x_label}$", y=f"${y_label}$")
        plotter.fig.colorbar(pcm, ax=plotter.ax, label=root_hist.GetZaxis().GetTitle())

        plotter.save( os.path.join(args.odir, Var2D) )

    #####################################
    # Plot grouped variables
    #####################################
    colors = hep.style.CMS['axes.prop_cycle'].by_key()['color']
    markers = ('o', 's', 'd')
    
    GroupedVarList = {
        'JetPt_EtaRegions': ('JetPt_B', 'JetPt_E', 'JetPt_F'),
        'GenPt_EtaRegions': ('GenPt_B', 'GenPt_E', 'GenPt_F'),
        'JetTypes_Pt': ('GenPt', 'JetPt', 'CorrJetPt'),
    }
    legend_labels = {'JetPt_B': "Barrel",  'GenPt_B': "Barrel",
                     'JetPt_E': "Endcap",  'GenPt_E': "Endcap",
                     'JetPt_F': "Forward", 'GenPt_F': "Forward",
                     'GenPt': 'GenPt', 'JetPt': 'JetPt', 'CorrJetPt': 'CorrJetPt'}
    
    for key, GroupedVar in GroupedVarList.items():
        plotter = Plotter(args.sample_label)
        
        for i_var, Var in enumerate(GroupedVar):
            root_hist = CheckRootFile(f"{dqm_dir}/{Var}")

            nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
            
            values = histo_values(root_hist)
            errors = histo_values(root_hist, errors=True)
            
            plt.errorbar(bin_centers, values, xerr=0.5 * bin_widths, yerr=errors, linestyle='', label=legend_labels[Var], color=colors[i_var], fmt=markers[i_var])
            plt.step(bin_edges[:-1], values, where="post", color=colors[i_var], linewidth=2)

        plotter.labels(x="$p_T\,$ [GeV]", y=f"# Jets", legend_title='')
        plotter.save( os.path.join(args.odir, key) )

    #####################################
    # Resolution vs pt from histograms
    #####################################

    pt_edges = (20., 30., 40., 100., 200., 300., 600.)
    pt_ranges = tuple(f"{int(pt_edges[i])}_{int(pt_edges[i+1])}" for i in range(len(pt_edges) - 1))
    widths = np.diff(pt_edges)
    centers = 0.5 * (np.array(pt_edges[:-1]) + np.array(pt_edges[1:]))

    v_res_type = []
    for i_res, ResType in enumerate(('PtRecoOverGen', 'PtCorrOverGen')):

        v_res_type_eta = []
        for EtaRegion in EtaInfo.regions():

            v_res_type_eta_pt = []

            for i_pt, PtRange in enumerate(pt_ranges):
                histo = CheckRootFile(f"{dqm_dir}/h_{ResType}_{EtaRegion}_Pt{PtRange}")
                mean = histo.GetMean()
                stddev = histo.GetRMS()
                v_res_type_eta_pt.append(stddev/mean if stddev != 0 else np.nan)
                
            v_res_type_eta.append(v_res_type_eta_pt)
        v_res_type.append(v_res_type_eta)

    plotter = Plotter(args.sample_label)
    for i_res, ResType in enumerate(('PtRecoOverGen', 'PtCorrOverGen')):
        for i_eta, EtaRegion in enumerate(EtaInfo.regions()):
            values = v_res_type[i_res][i_eta]
            eb1 = plotter.ax.errorbar(centers, values, xerr=0.5 * widths, fmt=EtaInfo.marker(EtaRegion), capsize=4, color=EtaInfo.color(EtaRegion))
            eb1[-1][0].set_linestyle('-' if i_res==0 else '--')

    plotter.ax.text(0.03, 0.97, f"{JetType}", transform=plotter.ax.transAxes, fontsize=fontsize,
                    verticalalignment='top', horizontalalignment='left')
    plotter.labels(x="Gen Jet $p_T$ [GeV]", y="$\sigma(p_{T}^{reco}/p_{T}^{gen})$ / $<p_{T}^{reco}/p_{T}^{gen}>$")
    plotter.limits(y=(0,0.6))

    from matplotlib.lines import Line2D
    # Legend for Eta regions (colored markers)
    eta_legend_elements = [
        Line2D([0], [0], color=EtaInfo.color(x), marker=EtaInfo.marker(x), linestyle='-', label=EtaInfo.label(x))
        for x in EtaInfo.regions()
    ]

    # Legend for response types (line styles)
    res_legend_elements = [
        Line2D([0], [0], color='grey', linestyle='-', label='Reco'),
        Line2D([0], [0], color='grey', linestyle='--', label='Corrected')
    ]

    legend_eta = plotter.ax.legend(handles=eta_legend_elements, loc='upper right', fontsize=fontsize)
    legend_res = plotter.ax.legend(handles=res_legend_elements, loc='upper right', fontsize=fontsize, bbox_to_anchor=(0.73, 0.99))
    plotter.ax.add_artist(legend_eta)

    plotter.save( os.path.join(args.odir, 'PtResolution_CorrVsReco') )
