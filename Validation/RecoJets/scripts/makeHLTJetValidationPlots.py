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

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

def CheckRootFile(hname, rebin=None):
    hist_orig = file.Get(hname)
    if not hist_orig:
        raise RuntimeError(f"WARNING: Histogram {hname} not found.")

    hist = hist_orig.Clone(hname + "_clone")
    hist.SetDirectory(0) # detach from file

    if rebin is not None:
        if isinstance(rebin, int) or isinstance(rebin, float):
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
                
    def save(self, name):
        for ext in self.extensions:
            print(" ### INFO: Saving " + name + '.' + ext)
            plt.savefig(name + '.' + ext)
        plt.close()

class HLabels:
    _eff_types = ('Efficiency', 'Fake Rate', 'Gen Duplicates Rate', 'Reco Duplicates Rate')
    _resol_types = ('RecoOverGen', 'CorrOverGen', 'CorrOverReco')
    
    def __init__(self, atype):
        assert (
            atype in self._eff_types or atype in self._resol_types
        ), f"Invalid type: {atype}"
        self.mytype = atype

    @staticmethod
    def eff_types():
        return HLabels._eff_types
    @staticmethod
    def resol_types():
        return HLabels._resol_types
    
    @property
    def ytitle(self):
        if self.mytype == 'Efficiency':
            return 'Jet Efficiency'
        elif self.mytype == 'Fake Rate':
            return 'Jet Fake Rate'
        elif self.mytype == 'Gen Duplicates Rate':
            return 'Jet Gen Duplicate Rate'
        elif self.mytype == 'Reco Duplicates Rate':
            return 'Jet Reco Duplicate Rate'
        elif self.mytype == 'RecoOverGen':
            return "$p_{T}^{reco}/p_{T}^{gen}\,$"
        elif self.mytype == 'CorrOverGen':
            return "$p_{T}^{corr}/p_{T}^{gen}\,$"
        elif self.mytype == 'CorrOverReco':
            return "$p_{T}^{corr}/p_{T}^{reco}\,$"
 
    @property
    def savename(self):
        return self.ytitle.replace(' ', '_')
    
    @property
    def xvars(self):
        return ('Eta', 'Phi', 'Pt')

    def nhisto(self, var, basedir): # numerator
        if self.mytype == 'Efficiency':
            name = basedir + '/MatchedGen' + var
        elif self.mytype == 'Fake Rate':
            name = basedir + '/MatchedJet' + var
        elif self.mytype == 'Gen Duplicates Rate':
            name = basedir + '/DuplicatesGen' + var
        elif self.mytype == 'Reco Duplicates Rate':
            name = basedir + '/DuplicatesJet' + var
        else:
            raise ValueError(f"Unavailable HLabels.nhisto for {self.mytype}")
        return name

    def dhisto(self, var, basedir): # denominator
        if self.mytype == 'Efficiency' or self.mytype == 'Gen Duplicates Rate':
            name = basedir + '/Gen' + var
        elif self.mytype == 'Fake Rate' or self.mytype == 'Reco Duplicates Rate':
            name = basedir + '/Jet' + var
        else:
            raise ValueError(f"Unavailable HLabels.nhisto for {self.mytype}")
        return name

    def rhisto(self, var, basedir): # ratio
        if self.mytype == 'Efficiency':
            name = basedir + '/Eff_vs_' + var
        elif self.mytype == 'Fake Rate':
            name = basedir + '/Fake_vs_' + var
        elif self.mytype == 'Gen Duplicates Rate':
            name = basedir + '/DupGen_vs_' + var
        elif self.mytype == 'Reco Duplicates Rate':
            name = basedir + '/Dup_vs_' + var
        else:
            raise ValueError(f"Unavailable HLabels.nhisto for {self.mytype}")
        return name
    
    def mhisto(self, var, basedir): # mean
        if self.mytype in self._resol_types:
            var_type = 'Gen' if 'Gen' in self.mytype else ''
            name = basedir + f'/Res_{self.mytype}_{var_type}{var}_Mean'
        else:
            raise ValueError(f"Unavailable HLabels.mhisto for {self.mytype}")
        return name

    def shisto(self, var, basedir): # sigma
        if self.mytype in self._resol_types:
            var_type = 'Gen' if 'Gen' in self.mytype else ''
            name = basedir + f'/Res_{self.mytype}_{var_type}{var}_Sigma'
        else:
            raise ValueError(f"Unavailable HLabels.shisto for {self.mytype}")
        return name

    def leglabel(self, isNum):
        HowMany = ''
        HowMany = '' if 'Duplicates' not in self.mytype else ' multiple'
        if self.mytype in ('Efficiency', 'Gen Duplicates Rate'):
            if isNum:
                label = f'Gen Jets $p_T > 20$ GeV matched to{HowMany} HLT jets'
            else:
                label = 'Gen Jets $p_T > 20$ GeV'
        elif self.mytype in ('Fake Rate', 'Reco Duplicates Rate'):
            if isNum:
                label = f'HLT Jets $p_T > 30$ GeV matched to{HowMany} gen jets'
            else:
                label = 'HLT Jets $p_T > 30$ GeV'
        return label

class EtaInfo:
    """
    Manage information related with the eta region.
    """
    info = {
        'B': ('black', 'o', r'$|\eta|<1.5$'),
        'E': ('red',   's', r'$1.5\leq|\eta|<3$'),
        'F': ('blue',  'd', r'$3\leq|\eta|<6$')
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
    tprofile_rebinning = {'B': (30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600), #barrel
                          'E': (30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600), # endcap
                          'F': (30, 40, 50, 60, 80, 120, 240, 600)} # forward

    if args.jet == 'hltAK4PFPuppiJets': JetType = "AK4 PF Puppi Jets"
    elif args.jet == 'hltAK4PFClusterJets': JetType = "AK4 PF Cluster Jets"
    elif args.jet == 'hltAK4PFJets': JetType = "AK4 PF Jets"
    elif args.jet == 'hltAK4PFCHSJets': JetType = "AK4 PF CHS Jets"
    else: JetType = args.jet

    #####################################
    # Plot 1D single variables
    #####################################

    # >>> Configurable
    Var1DList = {
        'HLT Jets':             'JetPt',
        'HLT Corrected Jets':   'CorrJetPt',
        'Gen-level Jets':       'GenPt',
    }
    for Label, Var in Var1DList.items():
        plotter = Plotter(args.sample_label)
        root_hist = CheckRootFile(f"{dqm_dir}/{Var}", rebin=None)
        nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
        values, errors = histo_values_errors(root_hist)

        plt.errorbar(bin_centers, values, xerr=0.5 * bin_widths, yerr=errors, linestyle='', fmt='s', color='black', linewidth=2, label=Label)
        plt.step(bin_edges[:-1], values, where="post", color='black')
        plotter.ax.text(0.03, 0.97, f"{JetType}", transform=plotter.ax.transAxes, fontsize=fontsize,
                        verticalalignment='top', horizontalalignment='left')

        plotter.limits(y=(0, 1.2*max(values)))
        plotter.labels(x="$p_T\,$ [GeV]", y=f"# Jets", legend_title='')
        plotter.save( os.path.join(args.odir, Var) )

    #####################################
    # Plot 2D single variables
    #####################################

    # >>> Configurable
    Var2DList = ('h2d_PtRecoOverGen_nCost_B', 'h2d_PtRecoOverGen_nCost_E', 'h2d_PtRecoOverGen_nCost_F', 
                 'h2d_PtRecoOverGen_chHad_B', 'h2d_PtRecoOverGen_chHad_E', 'h2d_PtRecoOverGen_chHad_F',
                 'h2d_PtRecoOverGen_neHad_B', 'h2d_PtRecoOverGen_neHad_E', 'h2d_PtRecoOverGen_neHad_F',
                 'h2d_PtRecoOverGen_chEm_B', 'h2d_PtRecoOverGen_chEm_E', 'h2d_PtRecoOverGen_chEm_F',
                 'h2d_PtRecoOverGen_neEm_B', 'h2d_PtRecoOverGen_neEm_E', 'h2d_PtRecoOverGen_neEm_F',)

    for Var2D in Var2DList:
        plotter = Plotter(args.sample_label, fontsize=15)
        root_hist = CheckRootFile(f"{dqm_dir}/{Var2D}")

        nbins_x, nbins_y, x_edges, y_edges = define_bins_2D(root_hist)
        values = histo_values_2D(root_hist)

        x_label = root_hist.GetXaxis().GetTitle().replace('#', '\\')
        y_label = root_hist.GetYaxis().GetTitle().replace('#', '\\')

        # Plot with mplhep's hist2d (preserves ROOT bin edges, color bar included)
        pcm = plotter.ax.pcolormesh(x_edges, y_edges, values, cmap='viridis', shading='auto')

        plotter.labels(x=f"${x_label}$", y=f"${y_label}$")
        plotter.fig.colorbar(pcm, ax=plotter.ax, label=root_hist.GetZaxis().GetTitle())

        plotter.save( os.path.join(args.odir, Var2D) )

    #####################################
    # Plot grouped variables
    #####################################
    
    # >>> Configurable
    GroupedVarList = {
        'JetPt_EtaRegions': {
            'Barrel':   'JetPt_B', 
            'Endcap':   'JetPt_E', 
            'Forward':  'JetPt_F',
        },
        'GenPt_EtaRegions': {
            'Barrel':   'GenPt_B', 
            'Endcap':   'GenPt_E', 
            'Forward':  'GenPt_F',
        },
        'JetTypes_Pt':  {
            'Gen-level Jets':       'GenPt', 
            'HLT Jets':             'JetPt', 
            'HLT Corrected Jets':   'CorrJetPt',
        },
    }

    colors = hep.style.CMS['axes.prop_cycle'].by_key()['color']
    markers = ('o', 's', 'd')

    for GroupedVar in GroupedVarList:
        plotter = Plotter(args.sample_label)
        
        for i_var, (Label, Var) in enumerate(GroupedVarList[GroupedVar].items()):
            root_hist = CheckRootFile(f"{dqm_dir}/{Var}", rebin=None)
            nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
            values, errors = histo_values_errors(root_hist)
            x_label = root_hist.GetXaxis().GetTitle().replace('#', '\\')
            y_label = root_hist.GetYaxis().GetTitle()

            plt.errorbar(bin_centers, values, xerr=0.5 * bin_widths, yerr=errors, linestyle='', label=Label, color=colors[i_var], fmt=markers[i_var])
            plt.step(bin_edges[:-1], values, where="post", color=colors[i_var], linewidth=2)

        plotter.labels(x=f"${x_label}$", y=y_label, legend_title='')
        plotter.save( os.path.join(args.odir, GroupedVar) )

    #####################################
    # Response
    #####################################

    for i_res, ResType in enumerate(('PtRecoOverGen', 'PtCorrOverGen', 'PtCorrOverReco')):

        for EtaRegion in EtaInfo.regions():

            MergedPtBins = {
                '20 < $p_T$ < 40 GeV'   : [f'h_{ResType}_{EtaRegion}_Pt20_30', f'h_{ResType}_{EtaRegion}_Pt30_40'],
                '40 < $p_T$ < 100 GeV'  : [f'h_{ResType}_{EtaRegion}_Pt40_100'],
                '100 < $p_T$ < 300 GeV' : [f'h_{ResType}_{EtaRegion}_Pt100_200', f'h_{ResType}_{EtaRegion}_Pt200_300'],
                '300 < $p_T$ < 600 GeV' : [f'h_{ResType}_{EtaRegion}_Pt300_600',],
                '600 < $p_T$ < 5000 GeV': [f'h_{ResType}_{EtaRegion}_Pt600_2000', f'h_{ResType}_{EtaRegion}_Pt2000_5000']
            }

            v_stacked_histo = []
            v_labels = []
            for i, (pt_label, hist_names) in enumerate(MergedPtBins.items()):
                axis = None
                stacked = None

                for j, hist_name in enumerate(hist_names):                    
                    root_hist = CheckRootFile(f"{dqm_dir}/{hist_name}", rebin=2)
                    
                    # root_hist = root_hist.Rebin(2) # Use for low statistics
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
                v_labels.append(pt_label)

            if len(v_stacked_histo) == 0: continue

            plotter = Plotter(args.sample_label)

            for stacked_histo, pt_label in zip(v_stacked_histo, v_labels):
                if stacked_histo.sum().value == 0:
                    print(f"WARNING: Skipping empty histogram for {pt_label}")
                    continue
                stacked_histo.plot(ax=plotter.ax, linewidth=2, label=pt_label, density=True)
                
            plotter.ax.text(0.03, 0.97, f"{JetType}\n{EtaInfo.label(EtaRegion)}", transform=plotter.ax.transAxes, fontsize=fontsize,
                            verticalalignment='top', horizontalalignment='left')

            plotter.labels(x="${}$".format(v_stacked_histo[0].axes[0].label),
                           y="# Jets",
                           legend_title=r"Jet $p_T$ range")

            plotter.save( os.path.join(args.odir, f"Response_{ResType}_{EtaRegion}") )
    
    #####################################
    # Scale and Resolution
    #####################################

    ResolOptions = {    # ymin, ymax, color
        'Scale'         : (0.0, 2.7, '#7a21dd'),
        'Resolution'    : (0.0, 0.8, '#5790fc')
    }

    for key in ResolOptions.keys():
        for resol_type in HLabels.resol_types():
            myResolLabel = HLabels(resol_type)
            for myXvar in myResolLabel.xvars:

                plotter = Plotter(args.sample_label)
                tprofile_mean = CheckRootFile( myResolLabel.mhisto(myXvar, dqm_dir), rebin=2)
                tprofile_sigma = CheckRootFile( myResolLabel.shisto(myXvar, dqm_dir), rebin=2)
                nbins, bin_edges, bin_centers, bin_widths = define_bins(tprofile_mean)
                means, mean_errors = histo_values_errors(tprofile_mean)
                sigmas, sigma_errors = histo_values_errors(tprofile_sigma)
                if key == 'Scale':
                    y, y_errors = means, mean_errors
                    ylabel = f"<{myResolLabel.ytitle}>"
                else:
                    y = [s / m if m != 0 else np.nan for s, m in zip(sigmas, means)]
                    y_errors = [np.sqrt((ds / m)**2 + ((s / m) * (dm / m))**2) if m != 0 else np.nan
                        for s, ds, m, dm in zip(sigmas, sigma_errors, means, mean_errors)]
                    ylabel = f"$\sigma$({myResolLabel.ytitle}) / <{myResolLabel.ytitle}>"

                plotter.ax.errorbar(bin_centers, y, xerr=0.5 * bin_widths, yerr=y_errors, linestyle='',
                                    fmt='o', color=ResolOptions[key][2], label=f'{key} {resol_type}')

                if 'Pt' not in myXvar:
                    xlabel = fr'$\{myXvar.lower()}$'
                else: 
                    xlabel = "$p_{T}^{gen}\,$ [GeV]" if 'Gen' in resol_type else "$p_{T}^{reco}\,$ [GeV]"
                
                plotter.labels(x=xlabel, y=ylabel, legend_title='')
                plotter.limits(y=(ResolOptions[key][0], ResolOptions[key][1]))
                Text = f"{JetType}\n{EtaInfo.label(myXvar[-1])}" if any(x in myXvar for x in ('_B', '_E', '_F')) else f"{JetType}"
                plotter.ax.text(0.03, 0.97, Text, transform=plotter.ax.transAxes,
                                fontsize=fontsize, verticalalignment='top', horizontalalignment='left')
                
                if key == 'Scale': 
                    plotter.ax.axhline(1.0, color='gray', linestyle='--', linewidth=2)

                plotter.save( os.path.join(args.odir, f'{key}_{resol_type}_{myXvar}') )
            
            plotter = Plotter(args.sample_label, fontsize=fontsize)
            for etareg in EtaInfo.regions():
                tprofile_mean = CheckRootFile( myResolLabel.mhisto(f"Pt_{etareg}", dqm_dir), rebin=tprofile_rebinning[etareg] )
                tprofile_sigma = CheckRootFile( myResolLabel.shisto(f"Pt_{etareg}", dqm_dir), rebin=tprofile_rebinning[etareg] )
                nbins, bin_edges, bin_centers, bin_widths = define_bins(tprofile_mean)
                means, mean_errors = histo_values_errors(tprofile_mean)
                sigmas, sigma_errors = histo_values_errors(tprofile_sigma)
                if key == 'Scale':
                    y, y_errors = means, mean_errors
                    ylabel = f"<{myResolLabel.ytitle}>"
                else:
                    y = [s / m if m != 0 else np.nan for s, m in zip(sigmas, means)]
                    y_errors = [np.sqrt((ds / m)**2 + ((s / m) * (dm / m))**2) if m != 0 else np.nan
                        for s, ds, m, dm in zip(sigmas, sigma_errors, means, mean_errors)]
                    ylabel = f"$\sigma$({myResolLabel.ytitle}) / <{myResolLabel.ytitle}>"
                
                plt.errorbar(bin_centers, y, xerr=0.5 * bin_widths, yerr=y_errors, linestyle='',
                            fmt=EtaInfo.marker(etareg), color=EtaInfo.color(etareg), label=EtaInfo.label(etareg))
                plt.step(bin_edges[:-1], y, where="post", color=EtaInfo.color(etareg))

            xlabel = "$p_{T}^{gen}\,$ [GeV]" if 'Gen' in resol_type else "$p_{T}^{reco}\,$ [GeV]"
            plotter.labels(x=xlabel, y=ylabel, legend_title='')
            plotter.limits(y=(ResolOptions[key][0], ResolOptions[key][1]))
            plotter.ax.text(0.03, 0.97, f"{JetType}", transform=plotter.ax.transAxes,
                            fontsize=fontsize, verticalalignment='top', horizontalalignment='left')
            plotter.save( os.path.join(args.odir, f'{key}_{resol_type}_Pt_EtaBins') )

    #####################################
    # Scale and Resolution Reco vs Corr
    #####################################

    for key in ResolOptions.keys():
        plotter = Plotter(args.sample_label, fontsize=fontsize)
        for i_res, resol_type in enumerate(['RecoOverGen', 'CorrOverGen']):
            myResolLabel = HLabels(resol_type)
            for etareg in EtaInfo.regions():
                tprofile_mean = CheckRootFile( myResolLabel.mhisto(f"Pt_{etareg}", dqm_dir), rebin=tprofile_rebinning[etareg] )
                tprofile_sigma = CheckRootFile( myResolLabel.shisto(f"Pt_{etareg}", dqm_dir), rebin=tprofile_rebinning[etareg] )
                nbins, bin_edges, bin_centers, bin_widths = define_bins(tprofile_mean)
                means, mean_errors = histo_values_errors(tprofile_mean)
                sigmas, sigma_errors = histo_values_errors(tprofile_sigma)
                if key == 'Scale':
                    y, y_errors = means, mean_errors
                    ylabel = f"<{myResolLabel.ytitle}>"
                else:
                    y = [s / m if m != 0 else np.nan for s, m in zip(sigmas, means)]
                    y_errors = [np.sqrt((ds / m)**2 + ((s / m) * (dm / m))**2) if m != 0 else np.nan
                        for s, ds, m, dm in zip(sigmas, sigma_errors, means, mean_errors)]
                    ylabel = f"$\sigma$({myResolLabel.ytitle}) / <{myResolLabel.ytitle}>"
                
                mfc = 'white' if i_res == 1 else EtaInfo.color(etareg)
                eb = plotter.ax.errorbar(bin_centers, y, xerr=0.5 * bin_widths, yerr=y_errors, linestyle='',
                            fmt=EtaInfo.marker(etareg), color=EtaInfo.color(etareg), label=EtaInfo.label(etareg), mfc=mfc)
                eb[-1][0].set_linestyle('-' if i_res==0 else '--')

        xlabel = "$p_{T}^{gen}\,$ [GeV]"
        plotter.labels(x=xlabel, y=ylabel)
        plotter.limits(y=(ResolOptions[key][0], ResolOptions[key][1]))
        plotter.ax.text(0.03, 0.97, f"{JetType}", transform=plotter.ax.transAxes,
                        fontsize=fontsize, verticalalignment='top', horizontalalignment='left')

        from matplotlib.lines import Line2D
        # Legend for Eta regions (colored markers)
        eta_legend_elements = [
            Line2D([0], [0], color=EtaInfo.color(x), marker=EtaInfo.marker(x), linestyle='-', label=EtaInfo.label(x))
            for x in EtaInfo.regions()
        ]

        # Legend for response types (line styles)
        res_legend_elements = [
            Line2D([0], [0], color='grey', marker='o', mfc='grey', linestyle='-', label='Reco'),
            Line2D([0], [0], color='grey', marker='o', mfc='white', linestyle='--', label='Corrected')
        ]

        legend_eta = plotter.ax.legend(handles=eta_legend_elements, loc='upper right', fontsize=fontsize)
        legend_res = plotter.ax.legend(handles=res_legend_elements, loc='upper right', fontsize=fontsize, bbox_to_anchor=(0.73, 0.99))
        plotter.ax.add_artist(legend_eta)
        plotter.save( os.path.join(args.odir, f'Pt{key}_CorrVsReco_New') )

    ########################################
    # Jet efficiency, fakes and duplicates
    ########################################

    eff_color = '#bd1f01'

    for eff_type in HLabels.eff_types():
        myEffLabel = HLabels(eff_type)
        for myXvar in myEffLabel.xvars:
            root_hist_num = CheckRootFile( myEffLabel.nhisto(myXvar, dqm_dir), rebin=2 )
            root_hist_den = CheckRootFile( myEffLabel.dhisto(myXvar, dqm_dir), rebin=2 )
            root_ratio = CheckRootFile( myEffLabel.rhisto(myXvar, dqm_dir), rebin=2 )

            nbins, bin_edges, bin_centers, bin_widths = define_bins(root_ratio)
            numerator_vals, _ = histo_values_errors(root_hist_num)
            denominator_vals, _ = histo_values_errors(root_hist_den)
            eff_values, eff_errors = histo_values_errors(root_ratio)

            plotter = Plotter(args.sample_label, grid_color=None, fontsize=fontsize)
            common_kwargs = dict(where="post", linewidth=2)
            plotter.ax.step(bin_edges[:-1], denominator_vals, label=myEffLabel.leglabel(isNum=False),  color="black", **common_kwargs)
            plotter.ax.step(bin_edges[:-1], numerator_vals, label=myEffLabel.leglabel(isNum=True), color="#9c9ca1", linestyle='-.', **common_kwargs)
            plotter.ax.fill_between(bin_edges[:-1], numerator_vals, step="post", alpha=0.3, color="#9c9ca1")

            label = root_hist_num.GetXaxis().GetTitle().replace('#', '\\')
            plotter.labels(x=f"${label}$", y="# Jets", legend_title='', legend_loc='upper left')
            plotter.limits(y=(0, 1.2*max(denominator_vals)))

            Text = f"{JetType}\n{EtaInfo.label(myXvar[-1])}" if any(x in myXvar for x in ('_B', '_E', '_F')) else f"{JetType}"
            plotter.ax.text(0.97, 0.97, Text, transform=plotter.ax.transAxes,
                            fontsize=fontsize, verticalalignment='top', horizontalalignment='right')

            ax2 = plotter.ax.twinx()
            ax2.errorbar(bin_centers, eff_values, xerr=0.5 * bin_widths, yerr=eff_errors, linestyle='', fmt='o', color=eff_color, label=eff_type)
            ax2.set_ylabel(myEffLabel.ytitle, color=eff_color)
            ax2.set_ylim(0,1.2)
            ax2.grid(color=eff_color, axis='y')
            ax2.tick_params(axis='y', labelcolor=eff_color)
            plotter.ax.grid(color=eff_color, axis='x')

            plotter.save( os.path.join(args.odir, myEffLabel.savename + '_' + myXvar) )

        plotter = Plotter(args.sample_label, fontsize=fontsize)
        for etareg in EtaInfo.regions():
            root_ratio = CheckRootFile( myEffLabel.rhisto(f"Pt_{etareg}", dqm_dir), rebin=tprofile_rebinning[etareg] )
            nbins, bin_edges, bin_centers, bin_widths = define_bins(root_ratio)
            eff_values, eff_errors = histo_values_errors(root_ratio)

            plt.errorbar(bin_centers, eff_values, xerr=0.5 * bin_widths, yerr=eff_errors, linestyle='',
                         fmt=EtaInfo.marker(etareg), color=EtaInfo.color(etareg), label=EtaInfo.label(etareg))
            # plt.step(bin_edges[:-1], eff_values, where="post", color=EtaInfo.color(etareg))

        plotter.ax.text(0.03, 0.97, f"{JetType}", transform=plotter.ax.transAxes,
                        fontsize=fontsize, verticalalignment='top', horizontalalignment='left')
        label = root_ratio.GetXaxis().GetTitle()
        plotter.labels(x=f"${label}$", y=myEffLabel.ytitle, legend_title='')
        if "Duplicates" in eff_type: 
            plotter.limits(y=(0.001,2), logY=True)
        else:
            plotter.limits(y=(0,1.25))

        plotter.save( os.path.join(args.odir, myEffLabel.ytitle.replace(' ', '_') + '_Pt_EtaBins') )
