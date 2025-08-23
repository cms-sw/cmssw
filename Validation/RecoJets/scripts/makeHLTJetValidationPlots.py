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
        if isinstance(rebin, int) or isinstance(rebin, float):
            hist = hist.Rebin(int(rebin), hname + "_rebin")
        elif hasattr(rebin, '__iter__'):
            bin_edges_c = array.array('d', rebin)
            hist = hist.Rebin(len(bin_edges_c) - 1, hname + "_rebin", bin_edges_c)
        else:
            raise ValueError(f"Unknown type for rebin: {type(rebin)}")

    return hist

def errorbar_declutter(eff, err, yaxmin, frac=0.01):
    """
    Filter uncertainties if they lie below the minimum (vertical) axis value.
    Used to plot the filtered points differently, for instance by displaying only the upper uncertainty.
    """
    filt = eff-err/2 <= yaxmin
    eff_filt = np.where(filt, np.nan, eff)
    err_filt = np.where(filt, np.nan, err)

    up_error = eff_filt + err_filt/2
    transform = blended_transform_factory(plotter.ax.transData, plotter.ax.transAxes)
    up_error = np.where(np.isnan(up_error), frac, up_error) # place at 0.9% above the minimum vertical axis value
    up_error = np.where(up_error != frac, np.nan, up_error)

    return eff_filt, err_filt, up_error, transform

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

    @staticmethod
    def fraction_label(labtype):
        return {
            'neHad':  'Neutral Hadron Energy Fraction',
            'chEm':   'Charged EM Energy Fraction',
            'chHad':  'Charged Hadron Energy Fraction',
            'neEm':   'Neutral EM Energy Fraction',
            'nConst': '# Jet Constituents',
        }[labtype]

    @staticmethod
    def pt_label(labtype, average=False):
        labels = {
            "pt"        : "$p_{T}\,$ [GeV]",
            "gen"       : "$p_{T}^{gen}\,$ [GeV]",
            "reco"      : "$p_{T}^{reco}\,$ [GeV]",
            "pt/gen"    : "$p_{T}/p_{T}^{gen}\,$",
            "reco/gen"  : "$p_{T}^{reco}/p_{T}^{gen}\,$",
            "corr/gen"  : "$p_{T}^{corr}/p_{T}^{gen}\,$",
            "corr/reco" : "$p_{T}^{corr}/p_{T}^{reco}\,$",
        }
        if average:
            return f'<' + labels[labtype] + '>'
        else:
            return labels[labtype]

    @staticmethod
    def resol_label(labtype):
        return f"$\sigma$({HLabels.pt_label(labtype)}) / {HLabels.pt_label(labtype, average=True)}"

    def ytitle(self, resol=False):
        return {'Efficiency'           : 'Jet Efficiency',
                'Fake Rate'            : 'Jet Fake Rate',
                'Gen Duplicates Rate'  : 'Jet Gen Duplicate Rate',
                'Reco Duplicates Rate' : 'Jet Reco Duplicate Rate',
                'RecoOverGen'          : HLabels.resol_label('reco/gen') if resol else HLabels.pt_label('reco/gen', average=True),
                'CorrOverGen'          : HLabels.resol_label('corr/gen') if resol else HLabels.pt_label('corr/gen', average=True),
                'CorrOverReco'         : HLabels.resol_label('corr/reco') if resol else HLabels.pt_label('corr/reco', average=True),
                }[self.mytype]
 
    @property
    def savename(self):
        return self.ytitle().replace(' ', '_')
    
    @property
    def xvars(self):
        return ('Eta', 'Phi', 'Pt')

    def nhisto(self, var, basedir):
        """Numerator"""
        return {'Efficiency'           : basedir + '/MatchedGen' + var,
                'Fake Rate'            : basedir + '/MatchedJet' + var,
                'Gen Duplicates Rate'  : basedir + '/DuplicatesGen' + var,
                'Reco Duplicates Rate' : basedir + '/DuplicatesJet' + var
                }[self.mytype]

    def dhisto(self, var, basedir):
        """Denominator"""
        return {'Efficiency'           : basedir + '/Gen' + var,
                'Fake Rate'            : basedir + '/Jet' + var,
                'Gen Duplicates Rate'  : basedir + '/Gen' + var,
                'Reco Duplicates Rate' : basedir + '/Jet' + var
                }[self.mytype]
                
    def rhisto(self, var, basedir):
        """Ratio"""
        return {'Efficiency': basedir + '/Eff_vs_' + var,
                'Fake Rate': basedir + '/Fake_vs_' + var,
                'Gen Duplicates Rate': basedir + '/DupGen_vs_' + var,
                'Reco Duplicates Rate': basedir + '/Dup_vs_' + var
                }[self.mytype]
    
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

    tprofile_rebinning = {'B': (30, 40, 50, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600), #barrel
                          'E': (30, 40, 50, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600), # endcap
                          'F': (30, 40, 50, 80, 120, 240, 600)} # forward

    if args.jet == 'hltAK4PFPuppiJets': JetType = "AK4 PF Puppi Jets"
    elif args.jet == 'hltAK4PFClusterJets': JetType = "AK4 PF Cluster Jets"
    elif args.jet == 'hltAK4PFJets': JetType = "AK4 PF Jets"
    elif args.jet == 'hltAK4PFCHSJets': JetType = "AK4 PF CHS Jets"
    else: JetType = args.jet

    colors = hep.style.CMS['axes.prop_cycle'].by_key()['color']
    markers = ('o', 's', 'd')
    errorbar_kwargs = dict(capsize=3, elinewidth=0.8, capthick=2, linewidth=2, linestyle='')

    #####################################
    # Plot 1D single variables
    #####################################

    Var1DList = {
        'HLT Jets'                            : 'JetPt',
        'HLT Corrected Jets'                  : 'CorrJetPt',
        'Gen-level Jets'                      : 'GenPt',
        'Photon Multiplicity'                 : 'photonMultiplicity',
        'Neutral Multiplicity'                : 'neutralMultiplicity',
        'Charged Multiplicity'                : 'chargedMultiplicity',
        'Neutral Hadron Multiplicity'         : 'neutralHadronMultiplicity',
        'Charged Hadron Multiplicity'         : 'chargedHadronMultiplicity',
    }
    
    for Label, Var in Var1DList.items():
        plotter = Plotter(args.sample_label)
        root_hist = CheckRootFile(f"{dqm_dir}/{Var}", rebin=None)
        nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
        values, errors = histo_values_errors(root_hist)

        plt.errorbar(bin_centers, values, xerr=None, yerr=errors,
                     fmt='s', color='black', label=Label, **errorbar_kwargs)
        plt.step(bin_edges[:-1], values, where="post", color='black')
        plotter.ax.text(0.03, 0.97, f"{JetType}", transform=plotter.ax.transAxes, fontsize=fontsize,
                        verticalalignment='top', horizontalalignment='left')

        if 'Multiplicity' in Var:
            plotter.limits(y=(5E-1, 2*max(values)), logY=True)
            plotter.labels(x=Label, y='# Jets')
        else:
            plotter.limits(y=(0, 1.2*max(values)), logY=False)
            plotter.labels(x=HLabels.pt_label('pt'), y='# Jets', legend_title='')

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
        plotter = Plotter(args.sample_label, fontsize=15)
        root_hist = CheckRootFile(f"{dqm_dir}/{Var2D}")

        nbins_x, nbins_y, x_edges, y_edges = define_bins_2D(root_hist)
        values = histo_values_2D(root_hist)

        y_label = root_hist.GetYaxis().GetTitle().replace('#', '\\')

        # Plot with mplhep's hist2d (preserves ROOT bin edges, color bar included)
        # empty bins will be invisible (background color)
        pcm = plotter.ax.pcolormesh(x_edges, y_edges, np.where(values==0, np.nan, values),
                                    cmap='viridis', shading='auto')

        if '_nCost_' in Var2D:
            xlabel = HLabels.fraction_label('nConst')
        elif '_chHad_' in Var2D:
            xlabel = HLabels.fraction_label('chHad')
        elif '_neHad_' in Var2D:
            xlabel = HLabels.fraction_label('neHad')
        elif '_chEm_' in Var2D:
            xlabel = HLabels.fraction_label('chEm')
        elif '_neEm_' in Var2D:
            xlabel = HLabels.fraction_label('neEm')
        else:
            raise RuntimeError(f'Label for variable {Var2D} is not supported.')

        plotter.labels(x=xlabel, y=f"${y_label}$")
        plotter.fig.colorbar(pcm, ax=plotter.ax, label='# Jets')

        plotter.save( os.path.join(args.odir, Var2D) )

    #####################################
    # Plot grouped variables
    #####################################
    
    GroupedVarList = dotdict({
        'JetPt_EtaRegions': dotdict(
            histos={'Barrel': 'JetPt_B', 'Endcap': 'JetPt_E', 'Forward': 'JetPt_F'},
            xlabel=HLabels.pt_label('pt'),
        ),
        'GenPt_EtaRegions': dotdict(
            histos={'Barrel': 'GenPt_B', 'Endcap': 'GenPt_E', 'Forward': 'GenPt_F'},
            xlabel=HLabels.pt_label('gen'),
        ),
        'JetTypes_Pt': dotdict(
            histos={'Gen-level Jets': 'GenPt', 'HLT Jets': 'JetPt', 'HLT Corrected Jets': 'CorrJetPt'},
            xlabel=HLabels.pt_label('pt'),
        ),
        'JetChargedHadFrac': dotdict(
            histos={'HLT jets': 'chargedHadronEnergyFraction',
                    'HLT jets matched': 'MatchedJetchHad'},
            xlabel=HLabels.fraction_label('chHad'),
        ),
        'JetNeutralHadFrac': dotdict(
            histos={'HLT jets':         'neutralHadronEnergyFraction',
                    'HLT jets matched': 'MatchedJetneHad'},
            xlabel=HLabels.fraction_label('neHad')
        ),
        'JetChargedEmFrac': dotdict(
            histos={'HLT jets':         'chargedEmEnergyFraction',
                    'HLT jets matched': 'MatchedJetchEm'},
            xlabel=HLabels.fraction_label('chEm')
        ),
        'JetNeutralEmFrac': dotdict(
            histos={'HLT jets': 'neutralEmEnergyFraction',
                    'HLT jets matched': 'MatchedJetneEm',},
            xlabel=HLabels.fraction_label('neEm')
        ),
        'JetnConst': dotdict(
            histos={'HLT jets':         'JetConstituents',
                    'HLT jets matched': 'MatchedJetnCost'},
            xlabel=HLabels.fraction_label('nConst')
        ),
        'photonMultiplicity': dotdict(
            histos={'Barrel': 'photonMultiplicity_B',
                    'Endcap': 'photonMultiplicity_E',
                    'Forward': 'photonMultiplicity_F'},
            xlabel="Photon Multiplicity"
        ),
        'neutralMultiplicity': dotdict(
            histos={'Barrel': 'neutralMultiplicity_B',
                    'Endcap': 'neutralMultiplicity_E',
                    'Forward': 'neutralMultiplicity_F'},
            xlabel="Neutral Multiplicity"
        ),
        'chargedMultiplicity': dotdict(
            histos={'Barrel': 'chargedMultiplicity_B',
                    'Endcap': 'chargedMultiplicity_E',
                    'Forward': 'chargedMultiplicity_F'},
            xlabel="Charged Multiplicity"
        ),
        'chargedHadronMultiplicity': dotdict(
            histos={'Barrel': 'chargedHadronMultiplicity_B',
                    'Endcap': 'chargedHadronMultiplicity_E',
                    'Forward': 'chargedHadronMultiplicity_F'},
            xlabel="Charged Hadron Multiplicity"
        ),
        'neutralHadronMultiplicity': dotdict(
            histos={'Barrel': 'neutralHadronMultiplicity_B',
                    'Endcap': 'neutralHadronMultiplicity_E',
                    'Forward': 'neutralHadronMultiplicity_F'},
            xlabel="Neutral Hadron Multiplicity"
        ),
    })

    for GroupedVar in GroupedVarList:
        plotter = Plotter(args.sample_label)
        
        for i_var, (Label, Var) in enumerate(GroupedVarList[GroupedVar]['histos'].items()):
            root_hist = CheckRootFile(f"{dqm_dir}/{Var}", rebin=None)
            nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
            values, errors = histo_values_errors(root_hist)

            plt.errorbar(bin_centers, values, xerr=None, yerr=errors,
                         label=Label, color=colors[i_var], fmt=markers[i_var], **errorbar_kwargs)
            plt.step(bin_edges[:-1], values, where="post", color=colors[i_var], linewidth=2)

        plotter.labels(x=GroupedVarList[GroupedVar].xlabel, y='# Jets' if 'Multiplicity' in GroupedVar else '# Jets', legend_title='')
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
                stacked_hist = None

                for j, hist_name in enumerate(hist_names):
                    rebin = 2 if EtaRegion != 'F' else 4
                    root_hist = CheckRootFile(f"{dqm_dir}/{hist_name}", rebin=rebin)
                        
                    if stacked_hist is None:
                        stacked_hist = root_hist.Clone()
                    else:
                        stacked_hist.Add(root_hist)

                if stacked_hist is None: continue
                v_stacked_histo.append(stacked_hist)
                v_labels.append(pt_label)

            if len(v_stacked_histo) == 0: continue

            plotter = Plotter(args.sample_label)
            for i, (stacked_histo, pt_label) in enumerate(zip(v_stacked_histo, v_labels)):
                if stacked_histo.Integral() == 0:
                    print(f"WARNING: Skipping empty histogram for {pt_label}")
                    continue
                if stacked_histo.Integral() < 2:
                    print(f"WARNING: Skipping histogram with low stat {pt_label}")
                    continue

                stacked_histo.Scale(1.0 / stacked_histo.Integral())
                nbins, bin_edges, bin_centers, bin_widths = define_bins(stacked_histo)
                values, errors = histo_values_errors(stacked_histo)

                plotter.ax.hist(bin_edges[:-1], bins=bin_edges, weights=values, histtype='stepfilled', color=colors[i], alpha=0.1)
                plotter.ax.hist(bin_edges[:-1], bins=bin_edges, weights=values, histtype='step', color=colors[i], linewidth=1.5)
                plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors, fmt='o', markersize=3,
                                    color=colors[i], label=pt_label, **errorbar_kwargs)
                
            plotter.ax.text(0.03, 0.97, f"{JetType}\n{EtaInfo.label(EtaRegion)}", transform=plotter.ax.transAxes, fontsize=fontsize,
                            verticalalignment='top', horizontalalignment='left')

            plotter.labels(x="${}$".format(v_stacked_histo[0].GetXaxis().GetTitle()),
                           y="[a.u.]",
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
                tprofile_mean = CheckRootFile( myResolLabel.mhisto(myXvar, dqm_dir), rebin=None)
                tprofile_sigma = CheckRootFile( myResolLabel.shisto(myXvar, dqm_dir), rebin=None)
                nbins, bin_edges, bin_centers, bin_widths = define_bins(tprofile_mean)
                means, mean_errors = histo_values_errors(tprofile_mean)
                sigmas, sigma_errors = histo_values_errors(tprofile_sigma)
                if key == 'Scale':
                    y, y_errors = means, mean_errors
                    ylabel = myResolLabel.ytitle(resol=False)
                else:
                    y = [s / m if m != 0 else np.nan for s, m in zip(sigmas, means)]
                    y_errors = [np.sqrt((ds / m)**2 + ((s / m) * (dm / m))**2) if m != 0 else np.nan
                        for s, ds, m, dm in zip(sigmas, sigma_errors, means, mean_errors)]
                    ylabel = myResolLabel.ytitle(resol=True) 

                plotter.ax.errorbar(bin_centers, y, xerr=None, yerr=y_errors,
                                    fmt='o', color=ResolOptions[key][2], label=f'{key} {resol_type}', **errorbar_kwargs)

                if 'Pt' not in myXvar:
                    xlabel = fr'$\{myXvar.lower()}$'
                else: 
                    xlabel = HLabels.pt_label('gen') if 'Gen' in resol_type else HLabels.pt_label('reco')
                
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
                    ylabel = myResolLabel.ytitle(resol=False)
                else:
                    y = [s / m if m != 0 else np.nan for s, m in zip(sigmas, means)]
                    y_errors = [np.sqrt((ds / m)**2 + ((s / m) * (dm / m))**2) if m != 0 else np.nan
                        for s, ds, m, dm in zip(sigmas, sigma_errors, means, mean_errors)]
                    ylabel = myResolLabel.ytitle(resol=True)
                
                plt.errorbar(bin_centers, y, xerr=None, yerr=y_errors,
                             fmt=EtaInfo.marker(etareg), color=EtaInfo.color(etareg), label=EtaInfo.label(etareg),
                             elinewidth=0.8, linewidth=2)
                plt.stairs(y, bin_edges, color=EtaInfo.color(etareg))

            xlabel = HLabels.pt_label('gen') if 'Gen' in resol_type else HLabels.pt_label('reco')
            plotter.labels(x=xlabel, y=ylabel, legend_title='')
            plotter.limits(y=(ResolOptions[key][0], ResolOptions[key][1]))
            plotter.ax.text(0.03, 0.97, f"{JetType}", transform=plotter.ax.transAxes,
                            fontsize=fontsize, verticalalignment='top', horizontalalignment='left')

            if key == 'Scale': 
                plotter.ax.axhline(1.0, color='gray', linestyle='--', linewidth=2)

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
                else:
                    y = [s / m if m != 0 else np.nan for s, m in zip(sigmas, means)]
                    y_errors = [np.sqrt((ds / m)**2 + ((s / m) * (dm / m))**2) if m != 0 else np.nan
                        for s, ds, m, dm in zip(sigmas, sigma_errors, means, mean_errors)]

                
                mfc = 'white' if i_res == 1 else EtaInfo.color(etareg)
                eb = plotter.ax.errorbar(bin_centers, y, xerr=0.5 * bin_widths, yerr=y_errors, mfc=mfc,
                                         fmt=EtaInfo.marker(etareg), color=EtaInfo.color(etareg), label=EtaInfo.label(etareg),
                                         elinewidth=0.8, linewidth=2)
                eb[-1][0].set_linestyle('-' if i_res==0 else '--') # horizontal erro bar

        if key == 'Scale':
            ylabel = HLabels.pt_label('pt/gen', average=True)
        else:
            ylabel = HLabels.resol_label('pt/gen')
        plotter.labels(x=HLabels.pt_label('gen'), y=ylabel)
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
        plotter.save( os.path.join(args.odir, f'Pt{key}_CorrVsReco') )

    ########################################
    # Jet efficiency, fakes and duplicates
    ########################################

    eff_color = '#bd1f01'
    for eff_type in HLabels.eff_types():
        myEffLabel = HLabels(eff_type)
        common_kwargs = dict(linestyle='', color=eff_color, label=eff_type)
        if any(x in myEffLabel.ytitle() for x in ('Fake', 'Duplicate')):
            if eff_type == 'Fake Rate':
                axmin = 1E-2
            else:
                axmin = 1E-4
            axmax = 2.4
        else:
            axmin, axmax = 0, 1.25

        for myXvar in myEffLabel.xvars:
            root_hist_num = CheckRootFile( myEffLabel.nhisto(myXvar, dqm_dir), rebin=2 )
            root_hist_den = CheckRootFile( myEffLabel.dhisto(myXvar, dqm_dir), rebin=2 )
            root_ratio = CheckRootFile( myEffLabel.rhisto(myXvar, dqm_dir), rebin=2 )

            nbins, bin_edges, bin_centers, bin_widths = define_bins(root_ratio)
            numerator_vals, _ = histo_values_errors(root_hist_num)
            denominator_vals, _ = histo_values_errors(root_hist_den)
            eff_values, eff_errors = histo_values_errors(root_ratio)
            if eff_type == 'Fake Rate':
                eff_values = np.array([1-i if i != 0 else np.nan for i in eff_values])

            plotter = Plotter(args.sample_label, grid_color=None, fontsize=fontsize)
            stepkwargs = dict(where="post", linewidth=2)
            plotter.ax.step(bin_edges[:-1], denominator_vals, label=myEffLabel.leglabel(isNum=False), color="black", **stepkwargs)
            plotter.ax.step(bin_edges[:-1], numerator_vals, label=myEffLabel.leglabel(isNum=True), color="#9c9ca1", linestyle='-.', **stepkwargs)
            plotter.ax.fill_between(bin_edges[:-1], numerator_vals, step="post", alpha=0.3, color="#9c9ca1")

            label = root_hist_num.GetXaxis().GetTitle().replace('#', '\\')
            plotter.labels(x=f"${label}$", y="# Jets", legend_title='', legend_loc='upper left')
            plotter.limits(y=(0, 1.2*max(denominator_vals)))

            Text = f"{JetType}\n{EtaInfo.label(myXvar[-1])}" if any(x in myXvar for x in ('_B', '_E', '_F')) else f"{JetType}"
            plotter.ax.text(0.97, 0.97, Text, transform=plotter.ax.transAxes,
                            fontsize=fontsize, verticalalignment='top', horizontalalignment='right')

            ax2 = plotter.ax.twinx()
            ax2.set_ylabel(myEffLabel.ytitle(), color=eff_color)

            if any(x in myEffLabel.ytitle() for x in ('Fake', 'Duplicate')):
                ax2.set_yscale('log')
            ax2.set_ylim(axmin, axmax)

            eff_filt, err_filt, up_error, transform = errorbar_declutter(eff_values, eff_errors, axmin)
            ax2.errorbar(bin_centers, eff_filt, xerr=0.5 * bin_widths, yerr=err_filt/2, fmt='o',
                         capthick=2, linewidth=1, capsize=2, **common_kwargs)
            ax2.plot(bin_centers, up_error, 'v', transform=transform, **common_kwargs)

            ax2.grid(color=eff_color, axis='y')
            ax2.tick_params(axis='y', labelcolor=eff_color)
            plotter.ax.grid(color=eff_color, axis='x')

            plotter.save( os.path.join(args.odir, myEffLabel.savename + '_' + myXvar) )

        plotter = Plotter(args.sample_label, fontsize=fontsize)
        for etareg in EtaInfo.regions():
            root_ratio = CheckRootFile( myEffLabel.rhisto(f"Pt_{etareg}", dqm_dir), rebin=tprofile_rebinning[etareg] )
            nbins, bin_edges, bin_centers, bin_widths = define_bins(root_ratio)
            eff_values, eff_errors = histo_values_errors(root_ratio)
            if eff_type == 'Fake Rate':
                eff_values = np.array([1-i if i != 0 else np.nan for i in eff_values])

            eff_filt, err_filt, up_error, transform = errorbar_declutter(eff_values, eff_errors, axmin)
            plotter.ax.errorbar(bin_centers, eff_filt, xerr=0.5 * bin_widths, yerr=err_filt/2,
                                fmt=EtaInfo.marker(etareg), color=EtaInfo.color(etareg), label=EtaInfo.label(etareg),
                                **errorbar_kwargs)
            plotter.ax.plot(bin_centers, up_error, 'v', linestyle='', color=EtaInfo.color(etareg), transform=transform)

        plotter.ax.text(0.03, 0.97, f"{JetType}", transform=plotter.ax.transAxes,
                        fontsize=fontsize, verticalalignment='top', horizontalalignment='left')
        label = root_ratio.GetXaxis().GetTitle()
        plotter.labels(x=f"${label}$", y=myEffLabel.ytitle(), legend_title='')
        if any(x in myEffLabel.ytitle() for x in ('Fake', 'Duplicate')):
            plotter.limits(y=(axmin, axmax), logY=True)
        else:
            plotter.limits(y=(0, axmax))

        plotter.save( os.path.join(args.odir, myEffLabel.ytitle().replace(' ', '_') + '_Pt_EtaBins') )
