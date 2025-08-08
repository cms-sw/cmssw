#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import hist
from hist import Hist
import array
import ROOT
import argparse
import os
import sys
import mplhep as hep
hep.style.use("CMS")

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make HLT Jet validation plots.')
    parser.add_argument('--file', type=str, required=True,               help='Paths to the DQM ROOT file.')
    parser.add_argument('--jet',  type=str, default='hltAK4PFPuppiJets', help='Name of the jet collection')
    parser.add_argument('--odir', type=str, required=False,              help='Path to the output directory (if not specified, save to current directory).')
    args = parser.parse_args()

    if args.odir:   OutDir = args.odir + '/'
    else:           OutDir = './'
    os.system(f'mkdir -p {OutDir}')

    file = ROOT.TFile.Open(args.file)
    dqm_dir = f"DQMData/Run 1/HLT/Run summary/JetMET/JetValidation/{args.jet}"
    if not file.Get(dqm_dir):
        sys.exit(f"Directory '{dqm_dir}' not found in {args.file}")

    fontsize = 18
    colors = ['black', 'red', 'blue']
    markers = ['o', 's', 'd']
    ResLabels = ['PtRecoOverGen', 'PtCorrOverGen', 'PtCorrOverReco']
    EtaLabels = ["$|\eta|<1.5$", "$1.5<|\eta|<3$", "$3<|\eta|<6$"]
    PtLabels = ["20 < $p_T$ < 40 GeV", "40 < $p_T$ < 100 GeV", "100 < $p_T$ < 300 GeV", "300 < $p_T$ < 600 GeV", "600 < $p_T$ < 5000 GeV"]

    tprofile_rebinning = [[30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600], # B
                          [30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 250, 300, 350, 400, 500, 600], # E
                          [30, 40, 50, 60, 80, 120, 240, 600]] # F
    
    def_colors = hep.style.CMS['axes.prop_cycle'].by_key()['color']

    if args.jet == 'hltAK4PFPuppiJets': JetType = "AK4 PF Puppi Jets"
    elif args.jet == 'hltAK4PFClusterJets': JetType = "AK4 PF Cluster Jets"
    elif args.jet == 'hltAK4PFJets': JetType = "AK4 PF Jets"
    elif args.jet == 'hltAK4PFCHSJets': JetType = "AK4 PF CHS Jets"
    else: JetType = args.jet

    #####################################
    # Response
    #####################################

    # '''
    for i_res, ResType in enumerate(['PtRecoOverGen', 'PtCorrOverGen', 'PtCorrOverReco']):

        for i_eta, EtaRegion in enumerate(['B', 'E', 'F']):

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
                    root_hist = file.Get(f"{dqm_dir}/{hist_name}")
                    if not root_hist:
                        print(f"WARNING: Histogram {dqm_dir}/{hist_name} not found.")
                        continue
                    
                    # root_hist = root_hist.Rebin(2) # [FIXME] due to low statistics
                    nbins = root_hist.GetNbinsX()
                    edges = [root_hist.GetBinLowEdge(i+1) for i in range(nbins)]
                    edges.append(root_hist.GetBinLowEdge(nbins+1))
                    values = np.array([root_hist.GetBinContent(i+1) for i in range(nbins)])
                    errors = np.array([root_hist.GetBinError(i+1) for i in range(nbins)])
                    label = root_hist.GetXaxis().GetTitle()

                    if axis is None:
                        axis = hist.axis.Variable(edges, name=root_hist.GetTitle(), label=label)

                    h = Hist(axis, storage=hist.storage.Weight())
                    h.view().value[:] = values
                    h.view().variance[:] = errors**2

                    if stacked is None:
                        stacked = h
                    else:
                        stacked += h

                if stacked == None: continue

                v_stacked_histo.append(stacked)

            if len(v_stacked_histo) == 0: continue

            fig, ax1 = plt.subplots(figsize=(10, 10))
            for i, stacked_histo in enumerate(v_stacked_histo):
                if stacked_histo.sum().value == 0:
                    print(f"WARNING: Skipping empty histogram for {PtLabels[i]}")
                    continue
                stacked_histo.plot(ax=ax1, linewidth=2, label=PtLabels[i])
            ax1.text(0.03, 0.97, f"{JetType}\n{EtaLabels[i_eta]}", transform=ax1.transAxes, fontsize=fontsize,
                verticalalignment='top', horizontalalignment='left')
            ax1.set_xlabel("${}$".format(v_stacked_histo[0].axes[0].label))
            ax1.set_ylabel("# Jets")
            ax1.grid(which='major', color='grey')
            ax1.legend(title="Jet $p_T$ range", title_fontsize=fontsize, fontsize=fontsize, loc='upper right')
            hep.cms.text(' Phase-2 Simulation Preliminary', fontsize=22, ax=ax1)
            hep.cms.lumitext("14 TeV", fontsize=22, ax=ax1)
            print(" ### INFO: Saving " + OutDir + f"Response{ResType}_{EtaRegion}.png")
            plt.savefig(OutDir + f"Response{ResType}_{EtaRegion}.png")
            plt.savefig(OutDir + f"Response{ResType}_{EtaRegion}.pdf")
            plt.close()
    # '''
    
    #####################################
    # Response vs pt from profile
    #####################################

    # '''
    x_axis_titles = ["$p_{T}^{gen}$", "$p_{T}^{gen}$", "$p_{T}^{reco}$"]
    y_axis_titles = ["$p_{T}^{reco}/p_{T}^{gen}$", "$p_{T}^{corr}/p_{T}^{gen}$", "$p_{T}^{corr}/p_{T}^{reco}$"]
    for i_res, ResType in enumerate(['PtRecoOverGen_GenPt', 'PtCorrOverGen_GenPt', 'PtCorrOverReco_Pt']):

        fig, ax1 = plt.subplots(figsize=(10, 10))
        for i_eta, EtaRegion in enumerate(['B', 'E', 'F']):
            tprofile = file.Get(f"{dqm_dir}/pr_{ResType}_{EtaRegion}")
            if not tprofile:
                print(f"WARNING: TProfile {dqm_dir}/pr_{ResType}_{EtaRegion} not found")
                continue

            bin_edges_c = array.array('d', tprofile_rebinning[i_eta])
            tprofile = tprofile.Rebin(len(tprofile_rebinning[i_eta]) - 1, "tprofile", bin_edges_c)
            nbins = tprofile.GetNbinsX()
            bin_edges = np.array([tprofile.GetBinLowEdge(i+1) for i in range(nbins)])
            bin_edges = np.append(bin_edges, tprofile.GetBinLowEdge(nbins+1))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_widths = np.diff(bin_edges)

            means = np.array([tprofile.GetBinContent(i+1) for i in range(nbins)])
            mean_errors = np.array([tprofile.GetBinError(i+1) for i in range(nbins)])

            ax1.errorbar(bin_centers, means, xerr=0.5 * bin_widths, yerr=mean_errors, linestyle='', fmt=markers[i_eta], color=colors[i_eta], label=EtaLabels[i_eta])
            ax1.step(bin_edges[:-1], means, where="post", color=colors[i_eta], linestyle='-', linewidth=2)

        ax1.set_xlabel(x_axis_titles[i_res]+" [GeV]")
        ax1.set_ylabel("<{}>".format(y_axis_titles[i_res]))
        ax1.text(0.03, 0.97, f"{JetType}", transform=ax1.transAxes, fontsize=fontsize,
            verticalalignment='top', horizontalalignment='left')
        ax1.legend(fontsize=fontsize, loc='upper right')
        ax1.set_ylim(0.0, 2.7)
        ax1.grid(which='major', color='grey')
        hep.cms.text(' Phase-2 Simulation Preliminary', fontsize=22, ax=ax1)
        hep.cms.lumitext("14 TeV", fontsize=22, ax=ax1)
        print(" ### INFO: Saving " + OutDir + f"Response_{ResType}.png")
        plt.savefig(OutDir + f"Response_{ResType}.png")
        plt.savefig(OutDir + f"Response_{ResType}.pdf")
        plt.close()
    # '''
    
    #####################################
    # Resolution vs pt from profile
    #####################################

    # '''
    names = ['pr_PtCorrOverGen_GenEta', 'pr_PtCorrOverGen_GenPt_B', 'pr_PtCorrOverGen_GenPt_E', 'pr_PtCorrOverGen_GenPt_F']

    x_axis_titles = ["$p_{T}^{gen}$", "$p_{T}^{gen}$", "$p_{T}^{reco}$"]
    y_axis_titles = ["$p_{T}^{reco}/p_{T}^{gen}$", "$p_{T}^{corr}/p_{T}^{gen}$", "$p_{T}^{corr}/p_{T}^{reco}$"]
    for i_res, ResType in enumerate(['PtRecoOverGen_GenPt', 'PtCorrOverGen_GenPt', 'PtCorrOverReco_Pt']):

        fig, ax1 = plt.subplots(figsize=(10, 10))
        for i_eta, EtaRegion in enumerate(['B', 'E', 'F']):
            tprofile = file.Get(f"{dqm_dir}/pr_{ResType}_{EtaRegion}")
            if not tprofile:
                print(f"WARNING: TProfile {dqm_dir}/pr_{ResType}_{EtaRegion} not found")
                continue

            bin_edges_c = array.array('d', tprofile_rebinning[i_eta][::2]) # [FIXME] due to low statistics
            tprofile = tprofile.Rebin(len(tprofile_rebinning[i_eta][::2]) - 1, "tprofile", bin_edges_c)
            nbins = tprofile.GetNbinsX()
            bin_edges = np.array([tprofile.GetBinLowEdge(i+1) for i in range(nbins)])
            bin_edges = np.append(bin_edges, tprofile.GetBinLowEdge(nbins+1))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_widths = np.diff(bin_edges)

            means = np.array([tprofile.GetBinContent(i+1) for i in range(nbins)])
            sigmas = np.array([tprofile.GetBinError(i+1) * np.sqrt(tprofile.GetBinEntries(i+1)) for i in range(nbins)])
            sigma_errors = np.array([tprofile.GetBinError(i+1) * np.sqrt(2)  for i in range(nbins)])
            resolution = [s / m if m != 0 else np.nan for s, m in zip(sigmas, means)]

            plt.errorbar(bin_centers, resolution, xerr=0.5 * bin_widths, yerr=sigma_errors, linestyle='', fmt=markers[i_eta], color=colors[i_eta], label=EtaLabels[i_eta])
            plt.step(bin_edges[:-1], resolution, where="post", color=colors[i_eta], linestyle='-', linewidth=2)

        ax1.set_xlabel(x_axis_titles[i_res]+" [GeV]")
        ax1.set_ylabel(f"$\sigma$({y_axis_titles[i_res]}) / <{y_axis_titles[i_res]}>")
        ax1.text(0.03, 0.97, f"{JetType}", transform=ax1.transAxes, fontsize=fontsize,
            verticalalignment='top', horizontalalignment='left')
        ax1.legend(fontsize=fontsize, loc='upper right')
        ax1.grid(which='major', color='grey')
        ax1.set_ylim(0,0.8)
        hep.cms.text(' Phase-2 Simulation Preliminary', fontsize=22, ax=ax1)
        hep.cms.lumitext("14 TeV", fontsize=22, ax=ax1)
        print(" ### INFO: Saving " + OutDir + f"Resolution_{ResType}.png")
        plt.savefig(OutDir + f"Resolution_{ResType}.png")
        plt.savefig(OutDir + f"Resolution_{ResType}.pdf")
        plt.close()
    # '''

    #####################################
    # Jet-Finding Efficiency
    #####################################

    den_label = ['HLT Jets $p_T > 30$ GeV', 'Gen Jets $p_T > 20$ GeV']
    num_label = ['HLT Jets $p_T > 30$ GeV matched to gen jets', 'Gen Jets $p_T > 20$ GeV matched to HLT jets']
    y_title = ['Jet Mistag Rate', 'Jet Efficiency']
    eff_color = '#bd1f01'

    # '''
    for i_type, (Type, Title) in enumerate(zip(['Jet', 'Gen'], ['JetMistagRate', 'JetEfficiency'])):
        for i_var, Var in enumerate(['Pt_B', 'Pt_E', 'Pt_F', 'Eta', 'Phi']):
            root_hist = file.Get(f"{dqm_dir}/{Type}{Var}")
            if not root_hist:
                print(f"WARNING: Histogram {dqm_dir}/{Type}{Var} not found.")
                continue

            root_hist_matched = file.Get(f"{dqm_dir}/Matched{Type}{Var}")
            if not root_hist_matched:
                print(f"WARNING: Histogram {dqm_dir}/Matched{Type}{Var} not found.")
                continue

            # root_hist.Rebin(3); root_hist_matched.Rebin(3) # [FIXME] due to very low statistics
            h_ratio = root_hist.Clone("h_ratio")
            h_ratio.Divide(root_hist_matched, root_hist, 1.0, 1.0, "B")

            nbins = h_ratio.GetNbinsX()
            bin_edges = np.array([h_ratio.GetBinLowEdge(i+1) for i in range(nbins)])
            bin_edges = np.append(bin_edges, h_ratio.GetBinLowEdge(nbins+1))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_widths = np.diff(bin_edges)

            denominator_vals = np.array([root_hist.GetBinContent(i+1) for i in range(nbins)])
            numerator_vals = np.array([root_hist_matched.GetBinContent(i+1) for i in range(nbins)])

            if i_type == 0: # JetMistagRate = 1 - Matched / Total
                eff_values = np.array([1 - h_ratio.GetBinContent(i+1) if h_ratio.GetBinContent(i+1) != 0 else 0 for i in range(nbins)])
            elif i_type == 1:
                eff_values = np.array([h_ratio.GetBinContent(i+1) for i in range(nbins)])
            eff_errors = np.array([h_ratio.GetBinError(i+1) for i in range(nbins)])
            label = root_hist.GetXaxis().GetTitle().replace('#', '\\')

            fig, ax1 = plt.subplots(figsize=(10, 10))
            ax1.step(bin_edges[:-1], denominator_vals, where="post", label=den_label[i_type], linewidth=2, color="black")
            ax1.step(bin_edges[:-1], numerator_vals, where="post", label=num_label[i_type], linewidth=2, color="#9c9ca1", linestyle='-.')
            ax1.fill_between(bin_edges[:-1], numerator_vals, step="post", alpha=0.3, color="#9c9ca1")
            ax1.set_xlabel(f"${label}$")
            ax1.set_ylabel("# Jets")
            ax1.set_ylim(0, 1.2*max(denominator_vals))
            ax1.grid(color=eff_color, axis='x')
            ax1.legend(fontsize=fontsize, loc='upper left')

            if 'Pt' in Var:
                ax1.text(0.97, 0.97, f"{EtaLabels[i_var]}", transform=ax1.transAxes, fontsize=fontsize,
                    verticalalignment='top', horizontalalignment='right')

            ax2 = ax1.twinx()
            ax2.errorbar(bin_centers, eff_values, xerr=0.5 * bin_widths, yerr=eff_errors, linestyle='', fmt='o', color=eff_color, label='Efficiency')
            ax2.set_ylabel(y_title[i_type], color=eff_color)
            ax2.set_ylim(0,1.2)
            ax2.grid(color=eff_color, axis='both')
            ax2.tick_params(axis='y', labelcolor=eff_color)
            hep.cms.text(' Phase-2 Preliminary', fontsize=fontsize, ax=ax1)
            hep.cms.lumitext("14 TeV", fontsize=fontsize, ax=ax1)
            print(" ### INFO: Saving " + OutDir + f"{Title}_{Var}.png")
            plt.savefig(OutDir + f"{Title}_{Var}.png")
            plt.savefig(OutDir + f"{Title}_{Var}.pdf")
            plt.close()

    for Type, Title in zip(['Jet', 'Gen'], ['JetMistagRate', 'JetEfficiency']):

        fig, ax1 = plt.subplots(figsize=(10, 10))
        for i_eta, EtaRegion in enumerate(['B', 'E', 'F']):
            root_hist = file.Get(f"{dqm_dir}/{Type}Pt_{EtaRegion}")
            if not root_hist:
                print(f"WARNING: Histogram {dqm_dir}/{Type}Pt_{EtaRegion} not found.")
                continue

            root_hist_matched = file.Get(f"{dqm_dir}/Matched{Type}Pt_{EtaRegion}")
            if not root_hist_matched:
                print(f"WARNING: Histogram {dqm_dir}/Matched{Type}Pt_{EtaRegion} not found.")
                continue

            # root_hist.Rebin(3); root_hist_matched.Rebin(3) # [FIXME] due to very low statistics
            h_ratio = root_hist.Clone("h_ratio")
            h_ratio.Divide(root_hist_matched, root_hist, 1.0, 1.0, "B")

            nbins = h_ratio.GetNbinsX()
            bin_edges = np.array([h_ratio.GetBinLowEdge(i+1) for i in range(nbins)])
            bin_edges = np.append(bin_edges, h_ratio.GetBinLowEdge(nbins+1))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_widths = np.diff(bin_edges)

            numerator_vals = np.array([root_hist_matched.GetBinContent(i+1) for i in range(nbins)])
            denominator_vals = np.array([root_hist.GetBinContent(i+1) for i in range(nbins)])

            if i_type == 0: # JetMistagRate = 1 - Matched / Total
                eff_values = np.array([1 - h_ratio.GetBinContent(i+1) if h_ratio.GetBinContent(i+1) != 0 else 0 for i in range(nbins)])
            elif i_type == 1:
                eff_values = np.array([h_ratio.GetBinContent(i+1) for i in range(nbins)])
            eff_errors = np.array([h_ratio.GetBinError(i+1) for i in range(nbins)])
            label = root_hist.GetXaxis().GetTitle()

            plt.errorbar(bin_centers, eff_values, xerr=0.5 * bin_widths, yerr=eff_errors, linestyle='', fmt=markers[i_eta], color=colors[i_eta], label=EtaLabels[i_eta])
            plt.step(bin_edges[:-1], eff_values, where="post", color=colors[i_eta])

        ax1.set_xlabel(f"${label}$")
        ax1.set_ylabel(y_title[i_type])
        ax1.set_ylim(0,1.25)
        ax1.grid(which='major', color='grey')
        ax1.legend(fontsize=fontsize, loc='upper right')
        hep.cms.text(' Phase-2 Preliminary', fontsize=fontsize, ax=ax1)
        hep.cms.lumitext("14 TeV", fontsize=fontsize, ax=ax1)
        print(" ### INFO: Saving " + OutDir + f"{Title}_Pt.png")
        plt.savefig(OutDir + f"{Title}_Pt.png")
        plt.savefig(OutDir + f"{Title}_Pt.pdf")
        plt.close()
    # '''

    #####################################
    # Plot 1D single variables
    #####################################

    # '''
    VarList = ['JetPt', 'GenPt', 'CorrJetPt']

    for Var in VarList:
        fig, ax1 = plt.subplots(figsize=(10, 10))
        root_hist = file.Get(f"{dqm_dir}/{Var}")
        if not root_hist:
            print(f"WARNING: Histogram {dqm_dir}/{Var} not found.")
            continue

        nbins = root_hist.GetNbinsX()
        bin_edges = np.array([root_hist.GetBinLowEdge(i+1) for i in range(nbins)])
        bin_edges = np.append(bin_edges, root_hist.GetBinLowEdge(nbins+1))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = np.diff(bin_edges)
        values = np.array([root_hist.GetBinContent(i+1) for i in range(nbins)])
        errors = np.array([root_hist.GetBinError(i+1) for i in range(nbins)])
        x_label = root_hist.GetXaxis().GetTitle().replace('#', '\\')
        y_label = root_hist.GetYaxis().GetTitle()

        plt.errorbar(bin_centers, values, xerr=0.5 * bin_widths, yerr=errors, linestyle='', fmt=markers[1], color=colors[1], linewidth=2)
        plt.step(bin_edges[:-1], values, where="post", color=colors[1])

        ax1.set_xlabel(f"${x_label}$")
        ax1.set_ylabel(f"{y_label}")
        ax1.grid(which='major', color='grey')
        hep.cms.text(' Phase-2 Preliminary', fontsize=fontsize, ax=ax1)
        hep.cms.lumitext("14 TeV", fontsize=fontsize, ax=ax1)
        print(" ### INFO: Saving " + OutDir + f"{Var}.png")
        plt.savefig(OutDir + f"{Var}.png")
        plt.savefig(OutDir + f"{Var}.pdf")
        plt.close()
    # '''

    #####################################
    # Plot 2D single variables
    #####################################

    # '''
    Var2DList = ['h2d_PtRecoOverGen_nCost_B', 'h2d_PtRecoOverGen_nCost_E', 'h2d_PtRecoOverGen_nCost_F', 
                 'h2d_PtRecoOverGen_chHad_B', 'h2d_PtRecoOverGen_chHad_E', 'h2d_PtRecoOverGen_chHad_F',
                 'h2d_PtRecoOverGen_neHad_B', 'h2d_PtRecoOverGen_neHad_E', 'h2d_PtRecoOverGen_neHad_F',
                 'h2d_PtRecoOverGen_chEm_B', 'h2d_PtRecoOverGen_chEm_E', 'h2d_PtRecoOverGen_chEm_F',
                 'h2d_PtRecoOverGen_neEm_B', 'h2d_PtRecoOverGen_neEm_E', 'h2d_PtRecoOverGen_neEm_F', ]

    for Var2D in Var2DList:
        fig, ax = plt.subplots(figsize=(10, 10))
        root_hist = file.Get(f"{dqm_dir}/{Var2D}")
        if not root_hist:
            print(f"WARNING: Histogram {dqm_dir}/{Var2D} not found.")
            continue

        nbins_x = root_hist.GetNbinsX()
        nbins_y = root_hist.GetNbinsY()

        x_edges = np.array([root_hist.GetXaxis().GetBinLowEdge(i+1) for i in range(nbins_x)])
        x_edges = np.append(x_edges, root_hist.GetXaxis().GetBinUpEdge(nbins_x))

        y_edges = np.array([root_hist.GetYaxis().GetBinLowEdge(j+1) for j in range(nbins_y)])
        y_edges = np.append(y_edges, root_hist.GetYaxis().GetBinUpEdge(nbins_y))

        # Fill values into a 2D numpy array (shape: ny x nx)
        values = np.array([
            [root_hist.GetBinContent(i+1, j+1) for i in range(nbins_x)]
            for j in range(nbins_y)
        ])

        x_label = root_hist.GetXaxis().GetTitle().replace('#', '\\')
        y_label = root_hist.GetYaxis().GetTitle().replace('#', '\\')

        # Plot with mplhep's hist2d (preserves ROOT bin edges, color bar included)
        pcm = ax.pcolormesh(x_edges, y_edges, values, cmap='viridis', shading='auto')

        # Axis labels and style
        ax.set_xlabel(f"${x_label}$")
        ax.set_ylabel(f"${y_label}$")
        fig.colorbar(pcm, ax=ax, label=root_hist.GetZaxis().GetTitle())

        hep.cms.text(' Phase-2 Preliminary', fontsize=fontsize, ax=ax)
        hep.cms.lumitext("14 TeV", fontsize=fontsize, ax=ax)

        print(" ### INFO: Saving " + OutDir + f"{Var2D}.png")
        plt.savefig(OutDir + f"{Var2D}.png")
        plt.savefig(OutDir + f"{Var2D}.pdf")
        plt.close()

    #####################################
    # Plot grouped variables
    #####################################

    GroupedVarList = {
        'JetPt_EtaRegions': ['JetPt_B', 'JetPt_E', 'JetPt_F'],
        'GenPt_EtaRegions': ['GenPt_B', 'GenPt_E', 'GenPt_F'],
        'JetTypes_Pt': ['GenPt', 'JetPt', 'CorrJetPt'],
    }

    for key, GroupedVar in GroupedVarList.items():

        fig, ax1 = plt.subplots(figsize=(10, 10))
        for i_var, Var in enumerate(GroupedVar):
            root_hist = file.Get(f"{dqm_dir}/{Var}")
            if not root_hist:
                print(f"WARNING: Histogram {dqm_dir}/{Var} not found.")
                continue

            nbins = h_ratio.GetNbinsX()
            bin_edges = np.array([h_ratio.GetBinLowEdge(i+1) for i in range(nbins)])
            bin_edges = np.append(bin_edges, h_ratio.GetBinLowEdge(nbins+1))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_widths = np.diff(bin_edges)
            values = np.array([root_hist.GetBinContent(i+1) for i in range(nbins)])
            errors = np.array([root_hist.GetBinError(i+1) for i in range(nbins)])
            x_label = root_hist.GetXaxis().GetTitle().replace('#', '\\')
            y_label = root_hist.GetYaxis().GetTitle()

            plt.errorbar(bin_centers, values, xerr=0.5 * bin_widths, yerr=errors, linestyle='', label=Var, color=def_colors[i_var], fmt=markers[i_var])
            plt.step(bin_edges[:-1], values, where="post", color=def_colors[i_var], linewidth=2)

        ax1.set_xlabel(f"${x_label}$")
        ax1.set_ylabel(f"{y_label}")
        ax1.grid(which='major', color='grey')
        ax1.legend(fontsize=fontsize, loc='upper right')
        hep.cms.text(' Phase-2 Preliminary', fontsize=fontsize, ax=ax1)
        hep.cms.lumitext("14 TeV", fontsize=fontsize, ax=ax1)
        print(" ### INFO: Saving " + OutDir + f"{key}.png")
        plt.savefig(OutDir + f"{key}.png")
        plt.savefig(OutDir + f"{key}.pdf")
        plt.close()

    #####################################
    # Resolution vs pt from histograms
    #####################################

    # '''
    pt_edges = [20., 30., 40., 100., 200., 300., 600.]
    pt_ranges = [f"{int(pt_edges[i])}_{int(pt_edges[i+1])}" for i in range(len(pt_edges) - 1)]
    widths = np.diff(pt_edges)
    centers = 0.5 * (np.array(pt_edges[:-1]) + np.array(pt_edges[1:]))

    v_res_type = []
    for i_res, ResType in enumerate(['PtRecoOverGen', 'PtCorrOverGen']):

        v_res_type_eta = []
        for i_eta, EtaRegion in enumerate(['B', 'E', 'F']):

            v_res_type_eta_pt = []

            for i_pt, PtRange in enumerate(pt_ranges):

                histo = file.Get(f"{dqm_dir}/h_{ResType}_{EtaRegion}_Pt{PtRange}")
                if not histo:
                    print(f"WARNING: Histogram {dqm_dir}/h_{ResType}_{EtaRegion}_Pt{PtRange} not found.")
                    v_res_type_eta_pt.append(np.nan)
                    continue

                mean = histo.GetMean()
                stddev = histo.GetRMS()
                if stddev != 0 :    v_res_type_eta_pt.append(stddev/mean)
                else:               v_res_type_eta_pt.append(np.nan)
            
            v_res_type_eta.append(v_res_type_eta_pt)

        v_res_type.append(v_res_type_eta)

    linestyles = ['-', '--']

    fig, ax1 = plt.subplots(figsize=(10, 10))
    for i_res, ResType in enumerate(['PtRecoOverGen', 'PtCorrOverGen']):
        for i_eta, EtaRegion in enumerate(['B', 'E', 'F']):
            values = v_res_type[i_res][i_eta]
            eb1 = ax1.errorbar(centers, values, xerr=0.5 * widths, fmt=markers[i_eta], capsize=4, color=colors[i_eta])
            eb1[-1][0].set_linestyle(linestyles[i_res])

    ax1.text(0.03, 0.97, f"{JetType}", transform=ax1.transAxes, fontsize=fontsize,
        verticalalignment='top', horizontalalignment='left')
    ax1.set_xlabel("Gen Jet $p_T$ [GeV]")
    ax1.set_ylabel("$\sigma(p_{T}^{reco}/p_{T}^{gen})$ / $<p_{T}^{reco}/p_{T}^{gen}>$")
    ax1.grid(which='major', color='grey')
    ax1.set_ylim(0,0.6)
    # ax1.set_xscale('log')

    from matplotlib.lines import Line2D
    # Legend for Eta regions (colored markers)
    eta_legend_elements = [
        Line2D([0], [0], color='black', marker=markers[0], linestyle='-', label=EtaLabels[0]),
        Line2D([0], [0], color='red',   marker=markers[1], linestyle='-', label=EtaLabels[1]),
        Line2D([0], [0], color='blue',  marker=markers[2], linestyle='-', label=EtaLabels[2])
    ]

    # Legend for response types (line styles)
    res_legend_elements = [
        Line2D([0], [0], color='grey', linestyle='-', label='Reco'),
        Line2D([0], [0], color='grey', linestyle='-', label='Corrected')
    ]

    legend_eta = ax1.legend(handles=eta_legend_elements, loc='upper right', fontsize=fontsize)
    legend_res = ax1.legend(handles=res_legend_elements, loc='upper right', fontsize=fontsize, bbox_to_anchor=(0.73, 0.99))
    ax1.add_artist(legend_eta)

    hep.cms.text(' Phase-2 Simulation Preliminary', fontsize=22, ax=ax1)
    hep.cms.lumitext("14 TeV", fontsize=22, ax=ax1)
    print(" ### INFO: Saving " + OutDir + f"PtResolution_CorrVsReco.png")
    plt.savefig(OutDir + f"PtResolution_CorrVsReco.png")
    plt.savefig(OutDir + f"PtResolution_CorrVsReco.pdf")
    plt.close()

    # '''