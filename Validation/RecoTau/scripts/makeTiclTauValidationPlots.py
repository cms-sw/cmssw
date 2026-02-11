#!/usr/bin/env python3

import os, sys, glob
import argparse
import numpy as np
import hist
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import array
import ROOT
import mplhep as hep
hep.style.use("CMS")

ROOT.gROOT.SetBatch(True)

# ======= CONFIG =======
PLOT_STEPS = [0, 1, 2, 3, 4, 5]
PLOT_ETA   = True              # enable eta plots

# Charged leg cap per DM (actual prongs) - only physical DMs, no DM 5
ch_legs_by_dm = {0:1, 1:1, 2:1, 10:3, 11:3}

# Photon leg cap per DM (only where pi0 exist); 2 photons per pi0
pho_legs_by_dm = {0:0, 1:2, 2:4, 10:0, 11:2}

# Pi0 cap per DM (for tau-level quantities)
pi0_cap_by_dm = {0:0, 1:1, 2:2, 10:0, 11:1}

colors_iter = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

dm_label_dict = {
    0: r"DM 0: $\tau^{\pm} \rightarrow \pi^{\pm} \nu_{\tau}$",
    1: r"DM 1: $\tau^{\pm} \rightarrow \pi^{\pm} \pi^{0} \nu_{\tau}$",
    2: r"DM 2: $\tau^{\pm} \rightarrow \pi^{\pm} \pi^{0} \pi^{0} \nu_{\tau}$",
    10: r"DM 10: $\tau^{\pm} \rightarrow \pi^{\pm} \pi^{\pm} \pi^{\pm} \nu_{\tau}$",
    11: r"DM 11: $\tau^{\pm} \rightarrow \pi^{\pm} \pi^{\pm} \pi^{\pm} \pi^{0} \nu_{\tau}$",
}

var_label_dict = {
    'pT': r'$p_{T}$ [GeV]',
    'eta': r'$\eta$',
}

step_label_dict = {
    0: "Step 0: CP→SimTrk",
    1: "Step 1: SimTrk→RecoTrk",
    2: "Step 2: RecoTrk→TICLCand",
    3: "Step 3: TICLCand→PF",
    4: "Step 4: Usage in a PFJet",
    5: "Step 5: Usage in a PFTau",
}

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

def draw_eff(obj, out):

    fontsize = 20
    fig, ax = plt.subplots(figsize=(10, 10))
    hep.cms.text(' Simulation Preliminary', ax=ax, fontsize=fontsize)
    hep.cms.lumitext(args.sample_label, ax=ax, fontsize=fontsize)
    
    nbins, bin_edges, bin_centers, bin_widths = define_bins(obj)
    values, errors = histo_values_errors(obj)

    ax.errorbar(bin_centers, values, xerr=None, yerr=errors, fmt='s', color=colors_iter[0], linewidth=2, markersize=8)
    ax.stairs(values, bin_edges, linewidth=2, baseline=None, color=colors_iter[0])
    
    # Extract variable from histogram name (pt or eta)
    hname = obj.GetName()
    if '_pt' in hname:
        ax.set_xlabel(r'$p_{T}(\tau)$ [GeV]', fontsize=fontsize)
    elif '_eta' in hname:
        ax.set_xlabel(r'$\eta(\tau)$', fontsize=fontsize)
    else:
        ax.set_xlabel('', fontsize=fontsize)
    
    ax.set_ylabel('Efficiency', fontsize=fontsize)
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    print(f'Saving plot: {out}')
    plt.savefig(out)
    plt.close()

def overlay_efficiency(list_objs, out):

    fontsize = 20
    fig, ax = plt.subplots(figsize=(10, 10))
    hep.cms.text(' Simulation Preliminary', ax=ax, fontsize=fontsize)
    hep.cms.lumitext(args.sample_label, ax=ax, fontsize=fontsize)

    for i,obj in enumerate(list_objs):
        nbins, bin_edges, bin_centers, bin_widths = define_bins(obj)
        values, errors = histo_values_errors(obj)

        label = obj.GetTitle().split(":")[0].strip()
        ax.errorbar(bin_centers, values, xerr=None, yerr=errors, fmt='s', label=label, color=colors_iter[i], linewidth=2, markersize=8)
        ax.stairs(values, bin_edges, linewidth=2, baseline=None, color=colors_iter[i])
    
    # Determine legend title from the actual histogram names (they may mix _all_ and _geNch_)
    import re
    legend_title = ""
    hnames = [obj.GetName() for obj in list_objs]
    
    # Check what types of histograms we have (priority: _all_ first)
    has_all = any('_all_' in h for h in hnames)
    has_ge_ch = any('_ge' in h and 'ch_' in h for h in hnames)
    has_ge_pi0 = any('_ge' in h and 'pi0_' in h for h in hnames)
    
    if has_all:
        legend_title = "Full tau efficiency"
    elif has_ge_ch:
        # Extract N from first _geNch histogram
        for h in hnames:
            if '_ge' in h and 'ch_' in h:
                match = re.search(r'_ge(\d+)ch', h)
                if match:
                    N = match.group(1)
                    legend_title = f"≥{N} charged hadron efficiency"
                    break
    elif has_ge_pi0:
        # Extract N from first _geNpi0 histogram
        for h in hnames:
            if '_ge' in h and 'pi0_' in h:
                match = re.search(r'_ge(\d+)pi0', h)
                if match:
                    N = match.group(1)
                    legend_title = f"≥{N} π⁰ efficiency"
                    break
                            
    # Extract variable from histogram name (pt or eta)
    hname = list_objs[0].GetName()
    if '_pt' in hname:
        ax.set_xlabel(r'$p_{T}(\tau)$ [GeV]', fontsize=fontsize)
    elif '_eta' in hname:
        ax.set_xlabel(r'$\eta(\tau)$', fontsize=fontsize)
    else:
        ax.set_xlabel('', fontsize=fontsize)
    ax.set_ylabel('Efficiency', fontsize=fontsize)
    ax.set_ylim([0, 1.1])
    ax.legend(title=legend_title, fontsize=fontsize-4, title_fontsize=fontsize-2, loc='best', frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()
    print(f'Saving plot: {out}')
    plt.savefig(out)
    plt.close()

def overlay_efficiency_with_gen(list_eff_objs, gen_obj, out):
    """Create efficiency plot with gen distribution on secondary y-axis"""
    fontsize = 20
    fig, ax1 = plt.subplots(figsize=(12, 10))
    hep.cms.text(' Simulation Preliminary', ax=ax1, fontsize=fontsize)
    hep.cms.lumitext(args.sample_label, ax=ax1, fontsize=fontsize)
    dm = int(list_eff_objs[0].GetTitle().split('DM ')[1].split(' ')[0])
    leg = int(list_eff_objs[0].GetTitle().split('leg')[1].split(' ')[0])
    var = list_eff_objs[0].GetTitle().split('vs ')[1]
    print(f"Plotting DM {dm}, leg {leg}, var {var}")

    # Detect particle type from histogram name
    hname = list_eff_objs[0].GetName()
    is_charged = 'eff_ch_' in hname
    is_photon = 'eff_pho_' in hname
    
    # Create leg label
    leg_labels = ['Leading', 'Subleading', 'Third', 'Fourth']
    leg_label = leg_labels[leg] if leg < len(leg_labels) else f'Leg {leg}'
    if is_charged:
        leg_title = f"{leg_label} hadron"
    elif is_photon:
        leg_title = f"{leg_label} photon"
    else:
        leg_title = f"{leg_label}"
    
    # Primary axis: efficiency curves
    for i, obj in enumerate(list_eff_objs):
        nbins, bin_edges, bin_centers, bin_widths = define_bins(obj)
        values, errors = histo_values_errors(obj)
        hname = obj.GetName()
        step_num = int(hname.split('_step')[1].split('_')[0])
        label = step_label_dict.get(step_num, f"Step {step_num}")
        ax1.errorbar(bin_centers, values, xerr=None, yerr=errors, fmt='s', 
                     label=label, color=colors_iter[i], linewidth=2, markersize=8)
        ax1.stairs(values, bin_edges, linewidth=2, baseline=None, color=colors_iter[i])
    
    ax1.set_ylabel('Efficiency', fontsize=fontsize, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim([0, 1.1])
    
    # Set x-axis label based on particle type
    if var == 'pT':
        if is_charged:
            xlabel = r'$p_{T}(\pi^{\pm})$ [GeV]'
        elif is_photon:
            xlabel = r'$p_{T}(\gamma)$ [GeV]'
        else:
            xlabel = r'$p_{T}$ [GeV]'
    else:  # eta
        if is_charged:
            xlabel = r'$\eta(\pi^{\pm})$'
        elif is_photon:
            xlabel = r'$\eta(\gamma)$'
        else:
            xlabel = r'$\eta$'
    ax1.set_xlabel(xlabel, fontsize=fontsize)

    # Secondary axis: gen distribution
    if gen_obj:
        ax2 = ax1.twinx()
        nbins_gen, bin_edges_gen, bin_centers_gen, bin_widths_gen = define_bins(gen_obj)
        values_gen, _ = histo_values_errors(gen_obj)
        
        # Plot as histogram with alpha transparency
        ax2.stairs(values_gen, bin_edges_gen, linewidth=2.5, baseline=None, 
                   color='gray', alpha=0.6, label='Gen distribution')
        ax2.bar(bin_centers_gen, values_gen, width=bin_widths_gen, 
                alpha=0.15, color='gray', edgecolor='none')
        
        ax2.set_ylabel('CP Entries', fontsize=fontsize, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

    if var == 'pT':
        ax1.legend(fontsize=fontsize-4, loc='lower right', bbox_to_anchor=(0.98, 0.15), 
                   title=f"{dm_label_dict[dm]}\n{leg_title}", title_fontsize=fontsize-2, 
                   frameon=True, fancybox=True, framealpha=0.6)
    else:
        ax1.legend(fontsize=fontsize-4, loc='upper center', 
                   title=f"{dm_label_dict[dm]}\n{leg_title}", title_fontsize=fontsize-2, 
                   frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()
    print(f'Saving plot: {out}')
    plt.savefig(out)
    plt.close()

def draw_hist(obj, out, opt=None):
    c = ROOT.TCanvas("c","c",900,650); obj.SetStats(0)
    if obj.InheritsFrom("TH2"): obj.Draw(opt or "COLZ")
    else:                        obj.Draw(opt or "HIST E1")
    c.SaveAs(out); c.Close()

def clone_conf_without_gen_dm5(h2):
    # Keep X as-is; rebuild Y without the bin whose label is "5"
    xax = h2.GetXaxis(); yax = h2.GetYaxis()
    x_labels = [xax.GetBinLabel(i) or str(i-1) for i in range(1, xax.GetNbins()+1)]
    y_labels = [yax.GetBinLabel(i) or str(i-1) for i in range(1, yax.GetNbins()+1)]
    keep_y = [(i, lab) for i, lab in enumerate(y_labels, start=1) if lab != "5"]
    ny = len(keep_y)
    # New hist with compact Y (no DM5)
    hnew = ROOT.TH2F(h2.GetName()+"_genNoDM5", h2.GetTitle()+";"+xax.GetTitle()+";"+yax.GetTitle()+" (no DM 5)",
                     len(x_labels), -0.5, len(x_labels)-0.5,
                     ny,           -0.5, ny-0.5)
    hnew.Sumw2()
    # Labels
    for ix, lab in enumerate(x_labels, start=1):
        hnew.GetXaxis().SetBinLabel(ix, lab)
    for jy_out, (_, lab) in enumerate(keep_y, start=1):
        hnew.GetYaxis().SetBinLabel(jy_out, lab)
    # Content copy (skip removed Y-bin)
    for ix in range(1, xax.GetNbins()+1):
        for jy_out, (jy_in, _) in enumerate(keep_y, start=1):
            val = h2.GetBinContent(ix, jy_in)
            err = h2.GetBinError(ix, jy_in)
            hnew.SetBinContent(ix, jy_out, val)
            hnew.SetBinError(ix, jy_out, err)
    return hnew

def draw_resolution(obj, out, xlabel):

    fontsize = 20
    fig, ax = plt.subplots(figsize=(10, 10))
    hep.cms.text(' Simulation Preliminary', ax=ax, fontsize=fontsize)
    hep.cms.lumitext(args.sample_label, ax=ax, fontsize=fontsize)

    nbins, bin_edges, bin_centers, bin_widths = define_bins(obj)
    values, errors = histo_values_errors(obj)

    ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                fmt='s', color=colors_iter[0], linewidth=2, markersize=8)
    ax.stairs(values, bin_edges, linewidth=2, baseline=None,
              color=colors_iter[0])

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel('Entries', fontsize=fontsize)
    # Extract DM from histogram name for title
    hname = obj.GetName()
    plt.tight_layout()
    print(f'Saving plot: {out}')
    plt.savefig(out)
    plt.close()

def overlay_resolution(list_objs, labels, out, xlabel, title=""):

    fontsize = 20
    fig, ax = plt.subplots(figsize=(10, 10))
    hep.cms.text(' Simulation Preliminary', ax=ax, fontsize=fontsize)
    hep.cms.lumitext(args.sample_label, ax=ax, fontsize=fontsize)

    for i, (obj, label) in enumerate(zip(list_objs, labels)):
        nbins, bin_edges, bin_centers, bin_widths = define_bins(obj)
        values, errors = histo_values_errors(obj)

        ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                    fmt='s', label=label, color=colors_iter[i % len(colors_iter)], linewidth=2, markersize=8)
        ax.stairs(values, bin_edges, linewidth=2, baseline=None,
                  color=colors_iter[i % len(colors_iter)])

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel('Entries', fontsize=fontsize)
    legend_kwargs = {'fontsize': fontsize-4, 'loc': 'best', 'frameon': True, 'fancybox': True, 'framealpha': 0.9}
    if title:
        legend_kwargs['title'] = title
        legend_kwargs['title_fontsize'] = fontsize
    ax.legend(**legend_kwargs)
    plt.tight_layout()
    print(f'Saving plot: {out}')
    plt.savefig(out)
    plt.close()

def overlay_hist(list_objs, labels, out, xlabel, ylabel, title=""):

    fontsize = 20
    fig, ax = plt.subplots(figsize=(10, 10))
    hep.cms.text(' Simulation Preliminary', ax=ax, fontsize=fontsize)
    hep.cms.lumitext(args.sample_label, ax=ax, fontsize=fontsize)

    for i, (obj, label) in enumerate(zip(list_objs, labels)):
        nbins, bin_edges, bin_centers, bin_widths = define_bins(obj)
        values, errors = histo_values_errors(obj)

        ax.errorbar(bin_centers, values, xerr=None, yerr=errors,
                    fmt='s', label=label, color=colors_iter[i % len(colors_iter)], linewidth=2, markersize=8)
        ax.stairs(values, bin_edges, linewidth=2, baseline=None,
                  color=colors_iter[i % len(colors_iter)])

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    legend_kwargs = {'fontsize': fontsize-4, 'loc': 'best', 'frameon': True, 'fancybox': True, 'framealpha': 0.9}
    if title:
        legend_kwargs['title'] = title
        legend_kwargs['title_fontsize'] = fontsize
    ax.legend(**legend_kwargs)
    plt.tight_layout()
    print(f'Saving plot: {out}')
    plt.savefig(out)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make Ticl Tau validation plots.')
    parser.add_argument('-s', '--step', type=str, default='HLT',                                   help='Validation step ("HLT" or "Offline")')
    parser.add_argument('-f', '--file', type=str, required=True,                                   help='Paths to the DQM ROOT file.')
    parser.add_argument('-o', '--odir', type=str, default="TauValidationPlots", required=False,    help='Path to the output directory.')
    parser.add_argument('-l', '--sample_label', type=str, default="Tau (200 PU)", required=False,  help='Sample label for plotting.')
    args = parser.parse_args()

    if args.step == 'HLT':
        dqm_dir = f"DQMData/Run 1/HLT/Run summary/TICL/ticlTauValidator"
        Step = 'HLT'
    elif args.step == 'Offline':
        dqm_dir = f"DQMData/Run 1/Run summary/RecoTauV/ticlTauValidator/"
        Step = 'offline'
    else:
        sys.exit("### ERROR: Please chose the step among the following ['HLT', 'Offline']")

    file = ROOT.TFile.Open(args.file)
    if not file or file.IsZombie():
        raise RuntimeError(f"Failed to open DQM file: {args.file}")
    if not file.Get(dqm_dir):
        raise RuntimeError(f"Directory '{dqm_dir}' not found in {args.file}")

    d = file.Get(dqm_dir)
    if not d:
        raise RuntimeError(f"Directory not found: {dqm_dir}")

    # List DM subdir
    dm_subdirs = []
    directory = file.GetDirectory(dqm_dir)
    for key in directory.GetListOfKeys():
        obj = key.ReadObj()
        if isinstance(obj, ROOT.TDirectory):
            name = obj.GetName()
            if name.startswith("GenDM"):
                dm_subdirs.append(name)

    print(dm_subdirs)
    dm_list = [int(dm_subdir.replace("GenDM", "")) for dm_subdir in dm_subdirs]

    # ---- Output dirs ----
    out_dir = args.odir
    os.makedirs(out_dir, exist_ok=True)

    missing = []

    # ======= 1) Per-leg, per-step efficiencies =======
    vars_to_plot = ["pt"] + (["eta"] if PLOT_ETA else [])
    for dm_subdir in dm_subdirs:
        dm = int(dm_subdir.replace("GenDM", ""))
        d_dm = file.Get(dqm_dir + "/" + dm_subdir)

        dm_dir = os.path.join(out_dir, f"dm{dm}")
        os.makedirs(dm_dir, exist_ok=True)

        # charged legs
        for L in range(ch_legs_by_dm[dm]):
            for var in vars_to_plot:
                list_eff_steps = []
                for S in PLOT_STEPS:
                    nm = f"eff_ch_dm{dm}_leg{L}_step{S}_{var}"
                    obj = d_dm.Get(nm)
                    if obj:
                        list_eff_steps.append(obj)
                    else:
                        print("MISSING:", f"{dqm_dir}/{dm_subdir}/{nm}")
                        missing.append(f"{dqm_dir}/{dm_subdir}/{nm}")
                
                # Try to get gen CP distribution for dual-axis plot
                gen_obj = d_dm.Get(f"cp_chHad_dm{dm}_{var}") if var == "pt" else d_dm.Get(f"cp_chHad_dm{dm}_{var}")
                
                overlay_efficiency_with_gen(
                    list_eff_steps,
                    gen_obj,
                    os.path.join(dm_dir, f"eff_ch_dm{dm}_leg{L}_{var}.png"),
                )

        # photon legs (only if DM has pi0)
        for L in range(pho_legs_by_dm[dm]):
            for var in vars_to_plot:
                list_eff_steps = []
                for S in PLOT_STEPS:
                    nm = f"eff_pho_dm{dm}_leg{L}_step{S}_{var}"
                    obj = d_dm.Get(nm)
                    if obj:
                        list_eff_steps.append(obj)
                    else:
                        print("MISSING:", f"{dqm_dir}/{dm_subdir}/{nm}")
                        missing.append(f"{dqm_dir}/{dm_subdir}/{nm}")
                
                # Try to get gen CP distribution for dual-axis plot
                gen_obj = d_dm.Get(f"cp_gamma_dm{dm}_{var}")
                
                overlay_efficiency_with_gen(
                    list_eff_steps,
                    gen_obj,
                    os.path.join(dm_dir, f"eff_pho_dm{dm}_leg{L}_{var}.png"),
                )

    # ======= 2) Tau-level efficiencies vs mother τ kinematics =======
    eff_all_pt_hists = []   # (label, hist)
    eff_all_eta_hists = []  # (label, hist)
    special_all_map = {0: 1, 10: 3}  # DM -> N for ≥N charged to include in "all" overlay
    eff_ge_ch_pt_hists = {}   # N -> list of hists
    eff_ge_ch_eta_hists = {}  # N -> list of hists
    for dm_subdir in dm_subdirs:
        dm = int(dm_subdir.replace("GenDM", ""))
        d_dm = file.Get(dqm_dir + "/" + dm_subdir)

        dm_dir = os.path.join(out_dir, f"dm{dm}")
        os.makedirs(dm_dir, exist_ok=True)

        # ≥N charged (N up to actual prongs)
        for N in range(1, ch_legs_by_dm[dm] + 1):
            for var in vars_to_plot:
                nm = f"eff_tau_dm{dm}_ge{N}ch_{var}"
                obj = d_dm.Get(nm)
                if obj:
                    draw_eff(obj, os.path.join(dm_dir, f"{nm}.png"))
                    if var == "pt":
                        eff_ge_ch_pt_hists.setdefault(N, []).append(obj)
                    elif var == "eta":
                        eff_ge_ch_eta_hists.setdefault(N, []).append(obj)
                else:
                    missing.append(f"{dqm_dir}/{dm_subdir}/{nm}")

        # ≥N pi0 (N up to pi0 count per DM)
        for N in range(1, pi0_cap_by_dm[dm] + 1):
            for var in vars_to_plot:
                nm = f"eff_tau_dm{dm}_ge{N}pi0_{var}"
                obj = d_dm.Get(nm)
                if obj:
                    draw_eff(obj, os.path.join(dm_dir, f"{nm}.png"))
                else:
                    missing.append(f"{dqm_dir}/{dm_subdir}/{nm}")

        # ALL expected (now for all DMs, including 0 and 10)
        for var in vars_to_plot:
            nm = f"eff_tau_dm{dm}_all_{var}"
            obj = d_dm.Get(nm)
            overlay_obj = obj
            if dm in special_all_map:
                geN = special_all_map[dm]
                alt_nm = f"eff_tau_dm{dm}_ge{geN}ch_{var}"
                alt_obj = d_dm.Get(alt_nm)
                if alt_obj:
                    overlay_obj = alt_obj
                elif not obj:
                    missing.append(f"{dqm_dir}/{dm_subdir}/{alt_nm}")

            if obj:
                draw_eff(obj, os.path.join(dm_dir, f"{nm}.png"))
            elif dm not in special_all_map and not overlay_obj:
                missing.append(f"{dqm_dir}/{dm_subdir}/{nm}")

            if overlay_obj:
                if var == "pt":
                    eff_all_pt_hists.append((f"DM {dm}", overlay_obj))
                elif var == "eta":
                    eff_all_eta_hists.append((f"DM {dm}", overlay_obj))

        # TAU-only split efficiencies (signal / iso)
        for N in range(1, ch_legs_by_dm[dm] + 1):
            for var in vars_to_plot:
                for kind in ["signal", "iso"]:
                    nm = f"eff_tau_dm{dm}_ge{N}ch_{kind}_{var}"
                    obj = d_dm.Get(nm)
                    if obj:
                        draw_eff(obj, os.path.join(dm_dir, f"{nm}.png"))
                    else:
                        missing.append(f"{dqm_dir}/{dm_subdir}/{nm}")

        for N in range(1, pi0_cap_by_dm[dm] + 1):
            for var in vars_to_plot:
                for kind in ["signal", "iso"]:
                    nm = f"eff_tau_dm{dm}_ge{N}pi0_{kind}_{var}"
                    obj = d_dm.Get(nm)
                    if obj:
                        draw_eff(obj, os.path.join(dm_dir, f"{nm}.png"))
                    else:
                        missing.append(f"{dqm_dir}/{dm_subdir}/{nm}")

    # Overlay final efficiencies (all expected) across DMs
    if eff_all_pt_hists:
        _, hists = zip(*eff_all_pt_hists)
        overlay_efficiency(
            list(hists),
            os.path.join(out_dir, "tau_eff_all_dm_overlay_pt.png"),
        )

    if PLOT_ETA and eff_all_eta_hists:
        _, hists = zip(*eff_all_eta_hists)
        overlay_efficiency(
            list(hists),
            os.path.join(out_dir, "tau_eff_all_dm_overlay_eta.png"),
        )

    # Overlay ≥N charged efficiencies across DMs (includes DM0 for N=1 and DM10 for N=3)
    for N, hists in sorted(eff_ge_ch_pt_hists.items()):
        overlay_efficiency(
            list(hists),
            os.path.join(out_dir, f"tau_eff_ge{N}ch_dm_overlay_pt.png"),
        )

    if PLOT_ETA:
        for N, hists in sorted(eff_ge_ch_eta_hists.items()):
            overlay_efficiency(
                list(hists),
                os.path.join(out_dir, f"tau_eff_ge{N}ch_dm_overlay_eta.png"),
            )

    # ======= 3) Inputs & matrices =======
    global_inputs_to_plot = [
        "cp_chHad_pt_all", "cp_chHad_eta_all",
        "cp_gamma_pt_all", "cp_gamma_eta_all",
        "dm_reco_vs_gen_jet",
        "dm_reco_vs_gen_tau",
        "dm_reco_vs_gen_hps",
    ]

    # final‑state labels (must match ticlTauValidator.cc)
    reco_labels = [
        "h^{#pm}",                # reco DM 0
        "h^{#pm}#pi^{0}",      # reco DM 1
        "h^{#pm}#pi^{0}#pi^{0}",      # reco DM 2
        "h^{#pm}h^{#pm}",                # reco DM 5
        "h^{#pm}h^{#pm}h^{#pm}",                # reco DM 10
        "h^{#pm}h^{#pm}h^{#pm}#pi^{0}",      # reco DM 11
    ]
    gen_labels  = [
        "h^{#pm}",                # gen DM 0
        "h^{#pm}#pi^{0}",      # gen DM 1
        "h^{#pm}#pi^{0}#pi^{0}",      # gen DM 2
        "h^{#pm}h^{#pm}h^{#pm}",                # gen DM 10
        "h^{#pm}h^{#pm}h^{#pm}#pi^{0}",      # gen DM 11
    ]

    for name in global_inputs_to_plot:
        obj = d.Get(name)
        if not obj:
            missing.append(f"{dqm_dir}/{name}")
            continue
        out = os.path.join(out_dir, f"{name}.png")
        if obj.InheritsFrom("TH2"):
            if name in ("dm_reco_vs_gen_jet", "dm_reco_vs_gen_tau", "dm_reco_vs_gen_hps"):
                xax = obj.GetXaxis()
                yax = obj.GetYaxis()
                for i, lab in enumerate(reco_labels, start=1):
                    if i <= xax.GetNbins():
                        xax.SetBinLabel(i, lab)
                for j, lab in enumerate(gen_labels, start=1):
                    if j <= yax.GetNbins():
                        yax.SetBinLabel(j, lab)

                if name == "dm_reco_vs_gen_jet":
                    obj.GetXaxis().SetTitle("Reco DM in jet")
                elif name == "dm_reco_vs_gen_tau":
                    obj.GetXaxis().SetTitle("Reco DM in tau")
                else:
                    obj.GetXaxis().SetTitle("DM assigned by HPS")
                obj.GetYaxis().SetTitle("Gen DM")

            draw_hist(obj, out, "COLZ")
        else:
            draw_hist(obj, out, "HIST E1")

    # ======= 4) Full tau-level histograms (raw inputs) =======
    # collect per-DM hists for global overlays
    gen_pt_hists = []   # list of (label, hist)
    reco_pt_hists = []  # list of (label, hist)
    gen_eta_hists = []  # list of (label, hist)
    reco_eta_hists = [] # list of (label, hist)

    for dm in dm_list:
        dm_dir = os.path.join(out_dir, f"dm{dm}")
        os.makedirs(dm_dir, exist_ok=True)

        d_dm = file.Get(f"{dqm_dir}/GenDM{dm}")
        if not d_dm:
            missing.append(f"{dqm_dir}/GenDM{dm}")
            continue

        # denominators
        for var in ["pt"] + (["eta"] if PLOT_ETA else []):
            nm = f"tau_dm{dm}_den_{var}"
            obj = d_dm.Get(nm)
            if obj:
                draw_hist(obj, os.path.join(dm_dir, f"{nm}.png"), "HIST E1")
            else:
                missing.append(f"{dqm_dir}/GenDM{dm}/{nm}")

        # ≥N charged numerators (N up to prongs)
        for N in range(1, ch_legs_by_dm[dm] + 1):
            for var in ["pt"] + (["eta"] if PLOT_ETA else []):
                nm = f"tau_dm{dm}_ge{N}ch_num_{var}"
                obj = d_dm.Get(nm)
                if obj:
                    draw_hist(obj, os.path.join(dm_dir, f"{nm}.png"), "HIST E1")
                else:
                    missing.append(f"{dqm_dir}/GenDM{dm}/{nm}")

        # ≥N pi0 numerators (N up to pi0 count per DM)
        for N in range(1, pi0_cap_by_dm[dm] + 1):
            for var in ["pt"] + (["eta"] if PLOT_ETA else []):
                nm = f"tau_dm{dm}_ge{N}pi0_num_{var}"
                obj = d_dm.Get(nm)
                if obj:
                    draw_hist(obj, os.path.join(dm_dir, f"{nm}.png"), "HIST E1")
                else:
                    missing.append(f"{dqm_dir}/GenDM{dm}/{nm}")

        # ALL expected numerators
        for var in ["pt"] + (["eta"] if PLOT_ETA else []):
            nm = f"tau_dm{dm}_all_num_{var}"
            obj = d_dm.Get(nm)
            if obj:
                draw_hist(obj, os.path.join(dm_dir, f"{nm}.png"), "HIST E1")
            else:
                missing.append(f"{dqm_dir}/GenDM{dm}/{nm}")

        # reco tau shapes
        for var in ["pt"] + (["eta"] if PLOT_ETA else []):
            nm = f"tau_dm{dm}_reco_{var}"
            obj = d_dm.Get(nm)
            if obj:
                draw_hist(obj, os.path.join(dm_dir, f"{nm}.png"), "HIST E1")
            else:
                missing.append(f"{dqm_dir}/GenDM{dm}/{nm}")

        # collect per-DM gen/reco hists for global DM overlays (pT)
        h_gen_pt = d_dm.Get(f"tau_dm{dm}_den_pt")
        if h_gen_pt:
            gen_pt_hists.append((f"DM {dm}", h_gen_pt))
        else:
            missing.append(f"{dqm_dir}/GenDM{dm}/tau_dm{dm}_den_pt")

        h_rec_pt = d_dm.Get(f"tau_dm{dm}_reco_pt")
        if h_rec_pt:
            reco_pt_hists.append((f"DM {dm}", h_rec_pt))
        else:
            missing.append(f"{dqm_dir}/GenDM{dm}/tau_dm{dm}_reco_pt")

        # collect per-DM gen/reco hists for global DM overlays (eta)
        if PLOT_ETA:
            h_gen_eta = d_dm.Get(f"tau_dm{dm}_den_eta")
            if h_gen_eta:
                gen_eta_hists.append((f"DM {dm}", h_gen_eta))
            else:
                missing.append(f"{dqm_dir}/GenDM{dm}/tau_dm{dm}_den_eta")

            h_rec_eta = d_dm.Get(f"tau_dm{dm}_reco_eta")
            if h_rec_eta:
                reco_eta_hists.append((f"DM {dm}", h_rec_eta))
            else:
                missing.append(f"{dqm_dir}/GenDM{dm}/tau_dm{dm}_reco_eta")

    # overlay of different DMs: one plot for gen and one for reco (pT)
    if gen_pt_hists:
        gen_labels, gen_hists = zip(*gen_pt_hists)
        overlay_hist(
            list(gen_hists),
            list(gen_labels),
            os.path.join(out_dir, "tau_gen_pt_dm_overlay.png"),
            r"$p_{T}(\tau)$ [GeV]",
            "Entries",
            "Gen $\\tau$ $p_{T}$"
        )

    if reco_pt_hists:
        reco_labels, reco_hists = zip(*reco_pt_hists)
        overlay_hist(
            list(reco_hists),
            list(reco_labels),
            os.path.join(out_dir, "tau_reco_pt_dm_overlay.png"),
            r"$p_{T}(\tau)$ [GeV]",
            "Entries",
            "Reco $\\tau$ $p_{T}$"
        )

    # optional: overlay of different DMs for eta (gen and reco)
    if PLOT_ETA:
        if gen_eta_hists:
            gen_eta_labels, gen_eta_objs = zip(*gen_eta_hists)
            overlay_hist(
                list(gen_eta_objs),
                list(gen_eta_labels),
                os.path.join(out_dir, "tau_gen_eta_dm_overlay.png"),
                r"$\eta(\tau)$",
                "Entries",
                "Gen $\\tau$ $\\eta$"
            )

        if reco_eta_hists:
            reco_eta_labels, reco_eta_objs = zip(*reco_eta_hists)
            overlay_hist(
                list(reco_eta_objs),
                list(reco_eta_labels),
                os.path.join(out_dir, "tau_reco_eta_dm_overlay.png"),
                r"$\eta(\tau)$",
                "Entries",
                "Reco $\\tau$ $\\eta$"
            )

    # ======= 5) pT resolution plots (pt_reco / pt_sim) =======
    HIST_NAME_PATTERN = "tau_dm{dm}_pt_reco_over_gen"

    res_hists = []  # (dm, hist)

    for dm in dm_list:
        dm_dir = os.path.join(out_dir, f"dm{dm}")
        os.makedirs(dm_dir, exist_ok=True)

        d_dm = file.Get(f"{dqm_dir}/GenDM{dm}")
        if not d_dm:
            missing.append(f"{dqm_dir}/GenDM{dm}")
            continue

        hname = HIST_NAME_PATTERN.format(dm=dm)
        h = d_dm.Get(hname)
        if not h:
            missing.append(f"{dqm_dir}/GenDM{dm}/{hname}")
            continue

        # individual DM plot
        out = os.path.join(dm_dir, f"{hname}.png")
        draw_resolution(h, out, r"$p_{T}^{reco} / p_{T}^{gen}$")

        res_hists.append((dm, h))

    # overlay all DMs in one plot
    if res_hists:
        objs   = [h for dm, h in res_hists]
        labels = [f"DM {dm}" for dm, h in res_hists]
        out_overlay = os.path.join(out_dir, "tau_pt_reco_over_gen_overlay.png")
        overlay_resolution(objs, labels, out_overlay,
                           r"$p_{T}^{reco}(\tau) / p_{T}^{gen}(\tau)$",
                           "$\\tau$ $p_{T}$ resolution")

    # ======= 6) Per-DM CP base histograms =======
    for dm in dm_list:
        dm_dir = os.path.join(out_dir, f"dm{dm}")
        os.makedirs(dm_dir, exist_ok=True)

        d_dm = file.Get(f"{dqm_dir}/GenDM{dm}")
        if not d_dm:
            missing.append(f"{dqm_dir}/GenDM{dm}")
            continue

        for nm in [
            f"cp_chHad_dm{dm}_pt",
            f"cp_chHad_dm{dm}_eta",
            f"cp_gamma_dm{dm}_pt",
            f"cp_gamma_dm{dm}_eta",
        ]:
            obj = d_dm.Get(nm)
            if obj:
                draw_hist(obj, os.path.join(dm_dir, f"{nm}.png"), "HIST E1")
            else:
                missing.append(f"{dqm_dir}/GenDM{dm}/{nm}")

    # ======= 7) CP-to-PF pT resolution (1D ratio histograms, per DM) =======
    cp_pf_had_hists = []  # (dm, hist) for overlays
    cp_pf_em_hists = []   # (dm, hist) for overlays

    for dm in dm_list:
        dm_dir = os.path.join(out_dir, f"dm{dm}")
        os.makedirs(dm_dir, exist_ok=True)

        d_dm = file.Get(f"{dqm_dir}/GenDM{dm}")
        if not d_dm:
            missing.append(f"{dqm_dir}/GenDM{dm}")
            continue

        # Hadronic (charged hadron) resolution
        obj_had = d_dm.Get(f"cp_pf_pt_resolution_hadronic_dm{dm}")
        if obj_had:
            draw_resolution(obj_had, os.path.join(dm_dir, f"cp_pf_pt_resolution_hadronic_dm{dm}.png"),
                           r"$p_{T}^{reco}/p_{T}^{gen}$")
            cp_pf_had_hists.append((dm, obj_had))
        else:
            missing.append(f"{dqm_dir}/GenDM{dm}/cp_pf_pt_resolution_hadronic_dm{dm}")

        # Electromagnetic (photon) resolution
        obj_em = d_dm.Get(f"cp_pf_pt_resolution_em_dm{dm}")
        if obj_em:
            draw_resolution(obj_em, os.path.join(dm_dir, f"cp_pf_pt_resolution_em_dm{dm}.png"),
                           r"$p_{T}^{reco}/p_{T}^{gen}$")
            cp_pf_em_hists.append((dm, obj_em))
        else:
            missing.append(f"{dqm_dir}/GenDM{dm}/cp_pf_pt_resolution_em_dm{dm}")

    # Create overlay plots of CP-to-PF resolution across DMs
    if cp_pf_had_hists:
        objs = [h for dm, h in cp_pf_had_hists]
        labels = [f'DM {dm}' for dm, h in cp_pf_had_hists]
        out_overlay_had = os.path.join(out_dir, "cp_pf_pt_resolution_hadronic_overlay.png")
        overlay_resolution(objs, labels, out_overlay_had, r"$p_{T}^{reco}(\pi^{\pm})/p_{T}^{gen}(\pi^{\pm})$",
                          "Hadronic CP resolution")

    if cp_pf_em_hists:
        objs = [h for dm, h in cp_pf_em_hists]
        labels = [f'DM {dm}' for dm, h in cp_pf_em_hists]
        out_overlay_em = os.path.join(out_dir, "cp_pf_pt_resolution_em_overlay.png")
        overlay_resolution(objs, labels, out_overlay_em, r"$p_{T}^{reco}(\gamma)/p_{T}^{gen}(\gamma)$",
                          "Gamma CP resolution")

    # ======= 8) Fake rate plots =======
    fake_dir = os.path.join(out_dir, "fake_rate")
    os.makedirs(fake_dir, exist_ok=True)

    # Fake rate histograms are in the FakeRate/ subfolder
    d_fake = file.Get(dqm_dir + "/FakeRate")
    if d_fake:
        # Inclusive fake rate (harvested efficiency = num/den)
        for var in vars_to_plot:
            nm = f"fake_rate_{var}"
            obj = d_fake.Get(nm)
            if obj:
                draw_eff(obj, os.path.join(fake_dir, f"{nm}.png"))
            else:
                missing.append(f"{dqm_dir}/FakeRate/{nm}")

        # Per reco-DM fake rate
        fake_dm_sel = [0, 1, 2, 5, 10, 11]
        fake_pt_hists = []   # for overlay
        fake_eta_hists = []  # for overlay
        for dm in fake_dm_sel:
            for var in vars_to_plot:
                nm = f"fake_rate_dm{dm}_{var}"
                obj = d_fake.Get(nm)
                if obj:
                    draw_eff(obj, os.path.join(fake_dir, f"{nm}.png"))
                    if var == "pt":
                        fake_pt_hists.append(obj)
                    elif var == "eta":
                        fake_eta_hists.append(obj)
                else:
                    missing.append(f"{dqm_dir}/FakeRate/{nm}")

        # Overlay per-DM fake rates
        if fake_pt_hists:
            overlay_efficiency(
                fake_pt_hists,
                os.path.join(fake_dir, "fake_rate_dm_overlay_pt.png"),
            )
        if PLOT_ETA and fake_eta_hists:
            overlay_efficiency(
                fake_eta_hists,
                os.path.join(fake_dir, "fake_rate_dm_overlay_eta.png"),
            )

        # Raw inputs: den/num histograms
        for var in vars_to_plot:
            for kind in ["den", "num"]:
                nm = f"fake_{kind}_{var}"
                obj = d_fake.Get(nm)
                if obj:
                    draw_hist(obj, os.path.join(fake_dir, f"{nm}.png"), "HIST E1")
                else:
                    missing.append(f"{dqm_dir}/FakeRate/{nm}")

        for dm in fake_dm_sel:
            for var in vars_to_plot:
                for kind in ["den", "num"]:
                    nm = f"fake_dm{dm}_{kind}_{var}"
                    obj = d_fake.Get(nm)
                    if obj:
                        draw_hist(obj, os.path.join(fake_dir, f"{nm}.png"), "HIST E1")
                    else:
                        missing.append(f"{dqm_dir}/FakeRate/{nm}")
    else:
        print(f"WARNING: FakeRate directory not found at {dqm_dir}/FakeRate")

    # Report missing without failing
    if missing:
        print("Missing objects:")
        for m in sorted(set(missing)):
            print("  -", m)

    file.Close()
    print(f"Done. Plots saved to: {out_dir}")
