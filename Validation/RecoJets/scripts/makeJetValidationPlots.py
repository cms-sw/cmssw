#!/usr/bin/env python3

import os, sys
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
        print(f"WARNING: Histogram {hname} not found.")
        return

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

def findBestGaussianCoreFit(histo, quiet=True):

    if not histo or histo.GetEntries() == 0:
        print(f"Histogram '{histo.GetName()}' is empty, skipping fit.")
        return None
    mean, rms = histo.GetMean(), histo.GetRMS()

    HalfMaxBinLow = histo.FindFirstBinAbove(histo.GetMaximum()/2)
    HalfMaxBinHigh = histo.FindLastBinAbove(histo.GetMaximum()/2)
    WidthAtHalfMaximum = 0.5*(histo.GetBinCenter(HalfMaxBinHigh) - histo.GetBinCenter(HalfMaxBinLow))
    Xmax = histo.GetXaxis().GetBinCenter(histo.GetMaximumBin())

    gausTF1 = ROOT.TF1()

    Pvalue = 0.
    RangeLow = histo.GetBinLowEdge(2)
    RangeUp = histo.GetBinLowEdge(histo.GetNbinsX())

    PvalueBest = 0.
    RangeLowBest = 0.
    RangeUpBest = 0.

    if Xmax < mean:
        meanForRange = Xmax
    else: # because some entries with LARGE errors sometimes falsely become the maximum
        meanForRange = mean

    if WidthAtHalfMaximum < rms and WidthAtHalfMaximum>0:
        spreadForRange = WidthAtHalfMaximum
    else: # WHF does not take into account weights and sometimes it turns LARGE
        spreadForRange = rms 

    rms_step_minus = 2.2
    while rms_step_minus>1.1:
        RangeLow = meanForRange - rms_step_minus*spreadForRange
        rms_step_plus = rms_step_minus

        while rms_step_plus>0.7:    
            RangeUp = meanForRange + rms_step_plus*spreadForRange
            if quiet:
                histo.Fit("gaus", "0Q", "0", RangeLow, RangeUp)
            else:
                histo.Fit("gaus", "0", "0", RangeLow, RangeUp)

            gausTF1 = histo.GetListOfFunctions().FindObject("gaus")
            if not gausTF1:
                print(f"Gaussian fit failed for '{histo.GetName()}'.")
                return
            ChiSquare = gausTF1.GetChisquare()
            ndf       = gausTF1.GetNDF()
            Pvalue = ROOT.TMath.Prob(ChiSquare, ndf)
            
            if Pvalue > PvalueBest:
                PvalueBest = Pvalue
                RangeLowBest = RangeLow
                RangeUpBest = RangeUp
                ndfBest = ndf
                ChiSquareBest = ChiSquare
                StepMinusBest = rms_step_minus
                StepPlusBest = rms_step_plus
                meanForRange = gausTF1.GetParameter(1)

            if not quiet:
                print("\n\nFitting range used: [Mean - " + str(rms_step_minus) + " sigma,  Mean + " + str(rms_step_plus) + " sigma ] ")
                print("ChiSquare = " + str(ChiSquare) + " NDF = " + str(ndf) + " Prob =  " + str(Pvalue) + "  Best Prob so far = " + str(PvalueBest))

            rms_step_plus = rms_step_plus - 0.1
            
        rms_step_minus = rms_step_minus - 0.1

    if quiet:
        histo.Fit("gaus", "0Q", "0", RangeLowBest, RangeUpBest)
    else:
        histo.Fit("gaus","0","0",RangeLowBest, RangeUpBest)
        print("\n\n\nMean =     " + str(mean) + "    Xmax = " + str(Xmax) + "  RMS = " + str(rms) + "  WidthAtHalfMax = " + str(WidthAtHalfMaximum))
        print("Fit found!")
        print("Final fitting range used: [Mean(Xmax) - " + str(StepMinusBest) + " rms(WHF), Mean(Xmax) + " + str(StepPlusBest) + " rms(WHF) ] ")
        print("ChiSquare = " + str(ChiSquareBest) + " NDF = " + str(ndfBest) + " Prob =     " + str(PvalueBest) + "\n\n")

    return histo.GetListOfFunctions().FindObject("gaus")

def tails_errors(n1, n2):
    """Compute and return the lower and upper errors."""
    if n1 == 0:
        return 0, 0
    # elif n1 < 0 or n2 < 0:
    #     raise RuntimeError(f"Negative number of entries! n1={n1}, n2={n2}")
    # elif n1 < n2:
    #     raise RuntimeError(f"n1 is smaller than n2! n1={n1}, n2={n2}")
    else:
        return ( n2/n1 - (n2/n1 + 0.5/n1 - np.sqrt(n2/pow(n1,2)*(1-n2/n1) + 0.25/pow(n1,2))) / (1+1.0/n1),
                 (n2/n1 + 0.5/n1 + np.sqrt(n2/pow(n1,2)*(1-n2/n1) + 0.25/pow(n1,2))) / (1+1.0/n1) - n2/n1 )

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
        assert atype in self._eff_types or atype in self._resol_types, f"Invalid type: {atype}"
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
            'neHad': 'Neutral Hadron Energy Fraction',
            'chEm':  'Charged EM Energy Fraction',
            'chHad': 'Charged Hadron Energy Fraction',
            'neEm':  'Neutral EM Energy Fraction',
            'nCost': '# Jet Constituents',
        }[labtype]

    @staticmethod
    def multiplicity_label(labtype):
        return {
            'neHad':  'Neutral Hadron Multiplicity',
            'chEm':   'Charged EM Multiplicity',
            'chHad':  'Charged Hadron Multiplicity',
            'neEm':   'Neutral EM Multiplicity',
            'chMult': 'Charged Multiplicity',
            'neMult': 'Neutral Multiplicity',
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
                label = f'Gen Jets $p_T > 20$ GeV matched to{HowMany} {Step} jets'
            else:
                label = 'Gen Jets $p_T > 20$ GeV'
        elif self.mytype in ('Fake Rate', 'Reco Duplicates Rate'):
            if isNum:
                label = f'{Step.capitalize()} Jets $p_T > 30$ GeV matched to{HowMany} gen jets'
            else:
                label = f'{Step.capitalize()} Jets $p_T > 30$ GeV'
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
    parser = argparse.ArgumentParser(description='Make Jet validation plots.')
    parser.add_argument('-s', '--step', type=str, default='HLT',                                   help='Validation step ("HLT" or "Offline")')
    parser.add_argument('-f', '--file', type=str, required=True,                                   help='Paths to the DQM ROOT file.')
    parser.add_argument('-j', '--jet',  type=str, default='hltAK4PFPuppiJets',                     help='Name of the jet collection')
    parser.add_argument('-o', '--odir', type=str, default="HLTJetValidationPlots", required=False, help='Path to the output directory.')
    parser.add_argument('-l', '--sample_label', type=str, default="QCD (200 PU)", required=False,  help='Sample label for plotting.')
    parser.add_argument('--doAllPlots', action='store_true', required=False,                       help='Run all plots. If not, only basic metrics.')
    args = parser.parse_args()
    
    if args.step == 'HLT':
        dqm_dir = f"DQMData/Run 1/HLT/Run summary/JetMET/JetValidation/{args.jet}"
        Step = 'HLT'
    elif args.step == 'Offline':
        dqm_dir = f"DQMData/Run 1/JetMET/Run summary/JetValidation/{args.jet}"
        Step = 'offline'
    else:
        sys.exit("### ERROR: Please chose the step among the following ['HLT', 'Offline']")
    
    file = ROOT.TFile.Open(args.file)
    if not file.Get(dqm_dir):
        raise RuntimeError(f"Directory '{dqm_dir}' not found in {args.file}")

    if not os.path.exists(args.odir):
        os.makedirs(args.odir)
    histo2d_dir = os.path.join(args.odir, 'histo2D')

    fontsize = 16       

    tprofile_rebinning = {'B': (30, 40, 50, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600), #barrel
                          'E': (30, 40, 50, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600), # endcap
                          'F': (30, 40, 50, 80, 120, 240, 600)} # forward

    JetType = {
        'hltAK4PFPuppiJets'   : "AK4 PF Puppi Jets",
        'hltAK4PFClusterJets' : "AK4 PF Cluster Jets",
        'hltAK4PFJets'        : "AK4 PF Jets",
        'hltAK4PFCHSJets'     : "AK4 PF CHS Jets",
        'ak4CaloJets'         : "AK4 CaloJets",
        'ak4PFJets'           : "AK4 PF Jets",
        'ak4PFJetsCHS'        : "AK4 PF CHS Jets",
        'slimmedJets'         : "Slimmed AK4 Jets",
        'slimmedJetsPuppi'    : "Slimmed AK4 PUPPI Jets",
        'slimmedJetsAK8'      : "Slimmed AK8 Jets",
    }.get(args.jet, args.jet)

    colors = hep.style.CMS['axes.prop_cycle'].by_key()['color']
    markers = ('o', 's', 'd')
    errorbar_kwargs = dict(capsize=3, elinewidth=0.8, capthick=2, linewidth=2, linestyle='')

    #####################################
    # Plot simple jet variables
    #####################################

    Var1DList = {
        f'{Step.capitalize()} Jets'           : 'JetPt',
        f'{Step.capitalize()} Corrected Jets' : 'CorrJetPt',
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
        if not root_hist: continue
        nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
        values, errors = histo_values_errors(root_hist)

        plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                     fmt='s', color='black', label=Label, **errorbar_kwargs)
        plotter.ax.step(bin_edges[:-1], values, where="post", color='black')
        plotter.ax.text(0.03, 0.97, f"{JetType}", transform=plotter.ax.transAxes, fontsize=fontsize,
                        verticalalignment='top', horizontalalignment='left')

        if 'Multiplicity' in Var:
            plotter.limits(y=(5E-1, 2*max(values)), logY=True)
            plotter.labels(x=Label, y='# Jets')
        else:
            plotter.limits(y=(0, 1.2*max(values)), logY=False)
            plotter.labels(x=HLabels.pt_label('pt'), y='# Jets', legend_title='')

        plotter.save( os.path.join(args.odir, Var) )

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

            eff_filt, (err_filt_lo, err_filt_hi), up_error, transform = rate_errorbar_declutter(eff_values, eff_errors, axmin)
            ax2.errorbar(bin_centers, eff_filt, xerr=0.5 * bin_widths, yerr=[err_filt_lo,err_filt_hi],
                         fmt='o', capthick=2, linewidth=1, capsize=2, **common_kwargs)
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

            eff_filt, (err_filt_lo, err_filt_hi), up_error, transform = rate_errorbar_declutter(eff_values, eff_errors, axmin)
            plotter.ax.errorbar(bin_centers, eff_filt, xerr=0.5 * bin_widths, yerr=[err_filt_lo,err_filt_hi],
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
    
    #####################################
    # Response distribution
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
                           y="Counts [a.u.]",
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
                
                plotter.ax.errorbar(bin_centers, y, xerr=None, yerr=y_errors,
                                    fmt=EtaInfo.marker(etareg), color=EtaInfo.color(etareg), label=EtaInfo.label(etareg),
                                    elinewidth=0.8, linewidth=2)
                plotter.ax.stairs(y, bin_edges, color=EtaInfo.color(etareg))

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

    if not args.doAllPlots:
        sys.exit()

    #####################################
    # Plot 2D single variables
    #####################################

    if not os.path.exists(histo2d_dir):
        os.makedirs(histo2d_dir)

    Var2DList = ('h2d_PtRecoOverGen_nCost_B', 'h2d_PtRecoOverGen_nCost_E', 'h2d_PtRecoOverGen_nCost_F', 
                 'h2d_PtRecoOverGen_chHad_B', 'h2d_PtRecoOverGen_chHad_E', 'h2d_PtRecoOverGen_chHad_F',
                 'h2d_PtRecoOverGen_neHad_B', 'h2d_PtRecoOverGen_neHad_E', 'h2d_PtRecoOverGen_neHad_F',
                 'h2d_PtRecoOverGen_chEm_B', 'h2d_PtRecoOverGen_chEm_E', 'h2d_PtRecoOverGen_chEm_F',
                 'h2d_PtRecoOverGen_neEm_B', 'h2d_PtRecoOverGen_neEm_E', 'h2d_PtRecoOverGen_neEm_F',
                 )

    for Var2D in Var2DList:
        plotter = Plotter(args.sample_label, fontsize=15)
        root_hist = CheckRootFile(f"{dqm_dir}/{Var2D}")

        nbins_x, nbins_y, x_edges, y_edges = define_bins_2D(root_hist)
        values = histo_values_2D(root_hist)

        ylabel = root_hist.GetYaxis().GetTitle().replace('#', '\\')

        # Plot with mplhep's hist2d (preserves ROOT bin edges, color bar included)
        # empty bins will be invisible (background color)
        pcm = plotter.ax.pcolormesh(x_edges, y_edges, np.where(values==0, np.nan, values),
                                    cmap='viridis', shading='auto')

        for lab2d in ('nCost', 'chHad', 'neHad', 'chEm', 'neEm'):
            if lab2d in Var2D:
                xlabel = HLabels.fraction_label(lab2d)
                    
        plotter.labels(x=xlabel, y=f"${ylabel}$")
        plotter.fig.colorbar(pcm, ax=plotter.ax, label='# Jets')

        plotter.save( os.path.join(histo2d_dir, Var2D) )

    ########################################
    # Tails
    ########################################

    nsigmas = (1, 2)
    ideal_fraction = {1: 0.3173, 2: 0.0455, 3: 0.0027}
    pt_bins = np.array((20, 30, 40, 60, 90, 150, 250, 400, 650, 1000))
    Var2DList = ('h2d_PtRecoOverGen_GenPt', )

    for Var2D in Var2DList:
        plotter = Plotter(args.sample_label, fontsize=15)
        root_hist = CheckRootFile(f"{dqm_dir}/{Var2D}") 

        tail_fracs, tail_low_errors, tail_high_errors = ({ns:[] for ns in nsigmas} for _ in range(3))
        for ipt, (low,high) in enumerate(zip(pt_bins[:-1],pt_bins[1:])):
            if root_hist.GetXaxis().FindBin(low) <= 0 and root_hist.GetXaxis().FindBin(high) >= root_hist.GetNbinsX():
                mess = f"Low bin: {low} | Low value: {root_hist.GetXaxis().FindBin(low)} | High bin: {high} | High value: {root_hist.GetXaxis().FindBin(high)}"
                raise RuntimeError(mess)

            hproj = root_hist.ProjectionY(root_hist.GetName() + "_proj" + str(ipt),
                                          root_hist.GetXaxis().FindBin(low), root_hist.GetXaxis().FindBin(high), "e")

            integr_start, integr_stop = 1, hproj.GetNbinsX()+1 # includes overflow
            integr = hproj.Integral(integr_start, integr_stop) 
            gausTF1 = findBestGaussianCoreFit(hproj)
            # plot individual projection + gaussian core fit
            plotter_single = Plotter(args.sample_label, fontsize=15)
            nbins, bin_edges, bin_centers, bin_widths = define_bins(hproj)
            values, errors = histo_values_errors(hproj)
            pt_label = str(low) + " < $p_T$ < " + str(high) + " GeV"
            plotter_single.ax.errorbar(bin_centers, values, xerr=None, yerr=errors/2, fmt='o', markersize=3,
                                       color='black', label=pt_label, **errorbar_kwargs)

            for ins, ns in enumerate(nsigmas):
                if not gausTF1:
                    print(f"Gaussian fit for '{root_hist.GetName()}' is empty, skipping plot.") 
                    tail_fracs[ns].append(np.nan)
                    tail_low_errors[ns].append(np.nan)
                    tail_high_errors[ns].append(np.nan)
                else:
                    xfunc = np.linspace(gausTF1.GetParameter(1) - ns*abs(gausTF1.GetParameter(2)), gausTF1.GetParameter(1) + ns*abs(gausTF1.GetParameter(2)))
                    yfunc = np.array([gausTF1.Eval(xi) for xi in xfunc])
                    plotter_single.ax.plot(xfunc, yfunc, 'o', color=colors[ins], linewidth=2, linestyle='--', label=f'{ns}$\sigma$ gaussian coverage')
        
                    tail_low = hproj.Integral(integr_start, hproj.FindBin(gausTF1.GetParameter(1) - ns*abs(gausTF1.GetParameter(2))))
                    tail_high = hproj.Integral(hproj.FindBin(gausTF1.GetParameter(1) + ns*abs(gausTF1.GetParameter(2))), integr_stop)
                    tail_fracs[ns].append((tail_low + tail_high) / integr)
                    tail_frac_errlo, tail_frac_errhi = tails_errors(integr, tail_low + tail_high)
                    tail_low_errors[ns].append(tail_frac_errlo)
                    tail_high_errors[ns].append(tail_frac_errhi)
            xlabel = hproj.GetXaxis().GetTitle().replace('#', '\\')
            plotter_single.labels(x=f"{xlabel}", y=f"# Jets", legend_title='')
            plotter_single.save( os.path.join(args.odir, Var2D  + '_tails_fit' + str(ipt)) )

        pt_centers = pt_bins[:-1] + (pt_bins[1:] - pt_bins[:-1])/2
        for ins, ns in enumerate(nsigmas):
            plotter.ax.errorbar(pt_centers, tail_fracs[ns], xerr=None, yerr=[tail_low_errors[ns],tail_low_errors[ns]],
                                fmt='o', markersize=3, color=colors[ins], **errorbar_kwargs)
            plotter.ax.stairs(tail_fracs[ns], pt_bins, color=colors[ins], linewidth=2)
            plotter.ax.axhline(y=ideal_fraction[ns], color=colors[ins], linestyle='--', label=f'{ns}$\sigma$ coverage')
        plotter.ax.set_xscale('log')
        xlabel = root_hist.GetXaxis().GetTitle().replace('#', '\\')
        plotter.labels(x=f"{xlabel}", y="Resolution Tail Fraction", legend_title='Ideal Gaussian')
        plotter.save( os.path.join(args.odir, Var2D  + '_tails') )
    
    ########################################
    # Multiplicities    
    ########################################

    Var2DList = (
        'h2d_chEm_pt_B', 'h2d_chEm_pt_E', 'h2d_chEm_pt_F',
        'h2d_neEm_pt_B', 'h2d_neEm_pt_E', 'h2d_neEm_pt_F',
        'h2d_chHad_pt_B', 'h2d_chHad_pt_E', 'h2d_chHad_pt_F',
        'h2d_neHad_pt_B', 'h2d_neHad_pt_E', 'h2d_neHad_pt_F',
        'h2d_chMult_pt_B', 'h2d_chMult_pt_E', 'h2d_chMult_pt_F',
        'h2d_neMult_pt_B', 'h2d_neMult_pt_E', 'h2d_neMult_pt_F',
        'h2d_chHadMult_pt_B', 'h2d_chHadMult_pt_E', 'h2d_chHadMult_pt_F',
        'h2d_neHadMult_pt_B', 'h2d_neHadMult_pt_E', 'h2d_neHadMult_pt_F',
        'h2d_phoMult_pt_B', 'h2d_phoMult_pt_E', 'h2d_phoMult_pt_F',
    )
    pt_bins = (20, 60, 100, 1000)
    
    for Var2D in Var2DList:
        root_hist = CheckRootFile(f"{dqm_dir}/{Var2D}")
        
        plotter = Plotter(args.sample_label, fontsize=15)

        nbins_x, nbins_y, x_edges, y_edges = define_bins_2D(root_hist)
        values = histo_values_2D(root_hist)

        xlabel = root_hist.GetXaxis().GetTitle().replace('#', '\\')

        # Plot with mplhep's hist2d (preserves ROOT bin edges, color bar included)
        # empty bins will be invisible (background color)
        pcm = plotter.ax.pcolormesh(x_edges, y_edges, np.where(values==0, np.nan, values),
                                    cmap='viridis', shading='auto')

        for lab2d in ('nCost', 'chHad', 'neHad', 'chEm', 'neEm', 'chMult', 'neMult'):
            if lab2d in Var2D:
                if 'Mult' in Var2D:
                    ylabel = HLabels.multiplicity_label(lab2d)
                else:
                    ylabel = HLabels.fraction_label(lab2d)
                break

        plotter.labels(x=f"${xlabel}$", y=ylabel)
        plotter.fig.colorbar(pcm, ax=plotter.ax, label='# Jets')
        plotter.save(os.path.join(histo2d_dir, Var2D))
        
        if 'Mult' not in Var2D:
            continue

        for ptbin in pt_bins:
            notfound = True
            for ibin in range(root_hist.GetNbinsX()+1):
                if root_hist.GetXaxis().GetBinLowEdge(ibin+1) == ptbin:
                    notfound = False
                    break
            if notfound:
                raise RuntimeError(f"The specified pT bin '{ptbin}' could not be matched to histogram {root_hist.GetName()}.")

        hproj = {}
        values_all, errors_all, labels_all = [], [], []
        nybins = root_hist.GetNbinsY()
        bin_edges = np.array(root_hist.GetXaxis().GetXbins())
        for ipt, (low,high) in enumerate(zip(pt_bins[:-1],pt_bins[1:])):
            plotter_single = Plotter(args.sample_label, fontsize=15)            
            hname = root_hist.GetName() + '_proj' + str(ipt)
            htitle = root_hist.GetTitle() + ' Proj PtBin' + str(ipt)
            if bin_edges.size:
                hproj[ipt] = ROOT.TH1F(hname, htitle, nybins, bin_edges)
            else:
                hproj[ipt] = ROOT.TH1F(hname, htitle, nybins,
                                       root_hist.GetXaxis().GetBinLowEdge(1), root_hist.GetXaxis().GetBinLowEdge(nybins+1))

            for ibin in range(root_hist.GetNbinsX()+1):
                xlow = root_hist.GetXaxis().GetBinLowEdge(ibin+1)
                xhigh = root_hist.GetXaxis().GetBinLowEdge(ibin+2)
                if low <= xlow and high >= xhigh:
                    for jbin in range(root_hist.GetNbinsX()+1):
                        # compute the weighted error between the current bin content and the one that will be added
                        val_curr = hproj[ipt].GetBinContent(jbin+1)
                        val_next = root_hist.GetBinContent(ibin+1, jbin+1)
                        error_curr = val_curr*hproj[ipt].GetBinError(jbin+1)
                        error_next = val_next*root_hist.GetBinError(ibin+1, jbin+1)
                        error = 0. if val_curr+val_next==0 else (error_curr + error_next) / (val_curr + val_next)
                        hproj[ipt].SetBinError(jbin+1, error)
                        hproj[ipt].SetBinContent(jbin+1, val_curr + val_next)
                        
            nbins, bin_edges, bin_centers, bin_widths = define_bins(hproj[ipt])
            values, errors = histo_values_errors(hproj[ipt])
            pt_label = str(low) + " < $p_T$ < " + str(high) + " GeV"
            values_all.append(values)
            errors_all.append(errors)
            labels_all.append(pt_label)
            plotter_single.ax.errorbar(bin_centers, values, xerr=None, yerr=errors/2, fmt='o', markersize=3,
                                       color='black', label=pt_label, **errorbar_kwargs)
            plotter_single.ax.stairs(values, bin_edges, color='black', linewidth=2)
            plotter_single.ax.set_yscale('log')
            plotter_single.ax.text(0.97, 0.92, f"{EtaInfo.label(Var2D[-1])}", transform=plotter_single.ax.transAxes, fontsize=fontsize,
                                   verticalalignment='top', horizontalalignment='right')
            plotter_single.labels(x=ylabel, y="# Jets", legend_title='')
            plotter_single.save(os.path.join(args.odir, Var2D + '_PtBin' + str(ipt)))

        plotter_all = Plotter(args.sample_label, fontsize=15)
        for idx, (vals, errs, lab) in enumerate(zip(values_all, errors_all, labels_all)):
            values = np.zeros_like(vals) if sum(vals)==0 else vals / sum(vals)
            errors = np.zeros_like(vals) if sum(vals)==0 else errs / sum(vals)
            plotter_all.ax.errorbar(bin_centers, values, xerr=None, yerr=errors/2, fmt='o', markersize=3,
                                    color=colors[idx], label=lab, **errorbar_kwargs)
            plotter_all.ax.stairs(values, bin_edges, color=colors[idx], linewidth=2)
        plotter_all.ax.set_yscale('log')
        plotter_all.ax.text(0.97, 0.79, f"{EtaInfo.label(Var2D[-1])}", transform=plotter_all.ax.transAxes, fontsize=fontsize,
                            verticalalignment='top', horizontalalignment='right')
        plotter_all.labels(x=ylabel, y="Counts [a.u.]", legend_title='Jet $p_T$ range')
        plotter_all.save(os.path.join(args.odir, Var2D + '_PtBinAll'))
        
    #####################################
    # Plot grouped variables
    #####################################
    
    GroupedVarList = dotdict({
        'JetPt_EtaRegions': dotdict(
            histos={'Barrel': 'JetPt_B', 
                    'Endcap': 'JetPt_E', 
                    'Forward': 'JetPt_F'},
            xlabel=HLabels.pt_label('pt'),
        ),
        'GenPt_EtaRegions': dotdict(
            histos={'Barrel': 'GenPt_B', 
                    'Endcap': 'GenPt_E', 
                    'Forward': 'GenPt_F'},
            xlabel=HLabels.pt_label('gen'),
        ),
        'JetTypes_Pt': dotdict(
            histos={'Gen-level Jets': 'GenPt', 
                    f'{Step.capitalize()} Jets': 'JetPt', 
                    f'{Step.capitalize()} Corrected Jets': 'CorrJetPt'},
            xlabel=HLabels.pt_label('pt'),
        ),
        'JetChargedHadFrac': dotdict(
            histos={f'{Step.capitalize()} jets': 'chargedHadronEnergyFraction',
                    f'{Step.capitalize()} jets matched': 'MatchedJetchHad'},
            xlabel=HLabels.fraction_label('chHad'),
        ),
        'JetNeutralHadFrac': dotdict(
            histos={f'{Step.capitalize()} jets':         'neutralHadronEnergyFraction',
                    f'{Step.capitalize()} jets matched': 'MatchedJetneHad'},
            xlabel=HLabels.fraction_label('neHad')
        ),
        'JetChargedEmFrac': dotdict(
            histos={f'{Step.capitalize()} jets':         'chargedEmEnergyFraction',
                    f'{Step.capitalize()} jets matched': 'MatchedJetchEm'},
            xlabel=HLabels.fraction_label('chEm')
        ),
        'JetNeutralEmFrac': dotdict(
            histos={f'{Step.capitalize()} jets': 'neutralEmEnergyFraction',
                    f'{Step.capitalize()} jets matched': 'MatchedJetneEm',},
            xlabel=HLabels.fraction_label('neEm')
        ),
        'JetnConst': dotdict(
            histos={f'{Step.capitalize()} jets':         'JetConstituents',
                    f'{Step.capitalize()} jets matched': 'MatchedJetnCost'},
            xlabel=HLabels.fraction_label('nCost')
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
            if not root_hist: continue
            nbins, bin_edges, bin_centers, bin_widths = define_bins(root_hist)
            values, errors = histo_values_errors(root_hist)

            plotter.ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                                label=Label, color=colors[i_var], fmt=markers[i_var], **errorbar_kwargs)
            plotter.ax.step(bin_edges[:-1], values, where="post", color=colors[i_var], linewidth=2)

        plotter.labels(x=GroupedVarList[GroupedVar].xlabel, y='# Jets' if 'Multiplicity' in GroupedVar else '# Jets', legend_title='')
        plotter.save( os.path.join(args.odir, GroupedVar) )