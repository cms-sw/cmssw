#!/usr/bin/env python3

import os, sys, re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import ROOT
import mplhep as hep
hep.style.use("CMS")

ROOT.gROOT.SetBatch(True)

# ======= CONFIG =======
PLOT_STEPS = [0, 1, 2, 3, 4, 5]
PLOT_TRACK_STEPS = [0, 1, 2, 3, 4]
PLOT_COMBI_STEPS = [0, 1, 2]
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

step_label_dict = {
    0: "Calo step 0: CP→SimTrk",
    1: "Calo step 1: SimTrk→RecoTrk",
    2: "Calo step 2: RecoTrk→TICLCand",
    3: "Calo step 3: TICLCand→PF",
    4: "Calo step 4: Usage in a PFJet",
    5: "Calo step 5: Usage in a PFTau",
}

track_step_label_dict = {
    0: "Track step 0: CP→TP",
    1: "Track step 1: TP→Track",
    2: "Track step 2: Track→PF",
    3: "Track step 3: PF→Jet",
    4: "Track step 4: Jet→Tau",
}

combi_step_label_dict = {
    0: "Combi step 0: Both→PF",
    1: "Combi step 1: Both→Jet",
    2: "Combi step 2: Both→Tau",
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
    
    legend_title = ""
    hnames = [obj.GetName() for obj in list_objs]
    
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

def overlay_efficiency_with_gen(list_eff_objs, gen_obj, out, step_labels=None):
    """Create efficiency plot with gen distribution on secondary y-axis"""
    if step_labels is None:
        step_labels = step_label_dict
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
        # Extract step number from _combistep, _trkstep, or _step
        m = re.search(r'_(combistep|trkstep|step)(\d+)', hname)
        step_num = int(m.group(2)) if m else 0
        label = step_labels.get(step_num, f"Step {step_num}")
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

def overlay_calo_and_track_chain(calo_objs, track_objs, gen_obj, out):
    """Overlay calo chain and track chain on the same plot.
    Calo chain: filled squares + solid lines.
    Track chain: open circles + dashed lines.
    """
    fontsize = 20
    fig, ax1 = plt.subplots(figsize=(14, 10))
    hep.cms.text(' Simulation Preliminary', ax=ax1, fontsize=fontsize)
    hep.cms.lumitext(args.sample_label, ax=ax1, fontsize=fontsize)

    ref_obj = calo_objs[0] if calo_objs else track_objs[0]
    dm = int(ref_obj.GetTitle().split('DM ')[1].split(' ')[0])
    leg = int(ref_obj.GetTitle().split('leg')[1].split(' ')[0])
    var = ref_obj.GetTitle().split('vs ')[1]

    leg_labels = ['Leading', 'Subleading', 'Third', 'Fourth']
    leg_label = leg_labels[leg] if leg < len(leg_labels) else f'Leg {leg}'

    # Calo chain
    for obj in calo_objs:
        nbins, bin_edges, bin_centers, bin_widths = define_bins(obj)
        values, errors = histo_values_errors(obj)
        hname = obj.GetName()
        m = re.search(r'_step(\d+)', hname)
        step_num = int(m.group(1)) if m else 0
        label = step_label_dict.get(step_num, f"Calo step {step_num}")
        col = colors_iter[step_num % len(colors_iter)]
        ax1.errorbar(bin_centers, values, xerr=None, yerr=errors, fmt='s',
                     label=label, color=col,
                     linewidth=2, markersize=8)
        ax1.stairs(values, bin_edges, linewidth=2, baseline=None, color=col)

    # Track chain — shift color index by +1 for steps >= 2
    for obj in track_objs:
        nbins, bin_edges, bin_centers, bin_widths = define_bins(obj)
        values, errors = histo_values_errors(obj)
        hname = obj.GetName()
        m = re.search(r'_trkstep(\d+)', hname)
        step_num = int(m.group(1)) if m else 0
        label = track_step_label_dict.get(step_num, f"Track step {step_num}")
        col_idx = step_num if step_num < 2 else step_num + 1
        col = colors_iter[col_idx % len(colors_iter)]
        ax1.errorbar(bin_centers, values, xerr=None, yerr=errors, fmt='o',
                     label=label, color=col,
                     linewidth=2, markersize=8, markerfacecolor='none', markeredgewidth=2)
        ax1.stairs(values, bin_edges, linewidth=2, baseline=None,
                   color=col, linestyle='--')

    ax1.set_ylabel('Efficiency', fontsize=fontsize, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim([0, 1.1])

    if var == 'pT':
        ax1.set_xlabel(r'$p_{T}(\pi^{\pm})$ [GeV]', fontsize=fontsize)
    else:
        ax1.set_xlabel(r'$\eta(\pi^{\pm})$', fontsize=fontsize)

    # Secondary axis: gen distribution
    if gen_obj:
        ax2 = ax1.twinx()
        nbins_gen, bin_edges_gen, bin_centers_gen, bin_widths_gen = define_bins(gen_obj)
        values_gen, _ = histo_values_errors(gen_obj)
        ax2.stairs(values_gen, bin_edges_gen, linewidth=2.5, baseline=None,
                   color='gray', alpha=0.6, label='Gen distribution')
        ax2.bar(bin_centers_gen, values_gen, width=bin_widths_gen,
                alpha=0.15, color='gray', edgecolor='none')
        ax2.set_ylabel('CP Entries', fontsize=fontsize, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

    chain_title = f"{dm_label_dict[dm]}\n{leg_label} hadron\nCalo (solid) vs Track (dashed)"
    if var == 'pT':
        ax1.legend(fontsize=fontsize-6, loc='lower right', bbox_to_anchor=(0.98, 0.05),
                   title=chain_title,
                   title_fontsize=fontsize-4, frameon=True, fancybox=True, framealpha=0.6,
                   ncol=2)
    else:
        ax1.legend(fontsize=fontsize-6, loc='upper center',
                   title=chain_title,
                   title_fontsize=fontsize-4, frameon=True, fancybox=True, framealpha=0.9,
                   ncol=2)
    plt.tight_layout()
    print(f'Saving plot: {out}')
    plt.savefig(out)
    plt.close()

def draw_hist(obj, out, opt=None):
    c = ROOT.TCanvas("c","c",900,650); obj.SetStats(0)
    if obj.InheritsFrom("TH2"): obj.Draw(opt or "COLZ")
    else:                        obj.Draw(opt or "HIST E1")
    c.SaveAs(out); c.Close()

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

args = None


class PlotContext:
    """Carries shared state across all plotting sections."""

    def __init__(self, root_file, dqm_dir, out_dir):
        self.file = root_file
        self.dqm_dir = dqm_dir
        self.out_dir = out_dir
        self.missing = []

        # Discover GenDM subdirectories
        directory = root_file.GetDirectory(dqm_dir)
        self.dm_subdirs = []
        for key in directory.GetListOfKeys():
            obj = key.ReadObj()
            if isinstance(obj, ROOT.TDirectory) and obj.GetName().startswith("GenDM"):
                self.dm_subdirs.append(obj.GetName())
        self.dm_list = [int(s.replace("GenDM", "")) for s in self.dm_subdirs]
        self.vars = ["pt"] + (["eta"] if PLOT_ETA else [])
        self.base_tdir = root_file.Get(dqm_dir)

    def get_dm_tdir(self, dm):
        return self.file.Get(f"{self.dqm_dir}/GenDM{dm}")

    def get_or_miss(self, tdir, name, parent_path=""):
        obj = tdir.Get(name) if tdir else None
        if not obj:
            path = f"{parent_path}/{name}" if parent_path else name
            self.missing.append(path)
        return obj

    def dm_out_dir(self, dm):
        d = os.path.join(self.out_dir, f"dm{dm}")
        os.makedirs(d, exist_ok=True)
        return d


def _plot_chain_effs(ctx, d_dm, dm, dm_dir,
                     particle, n_legs, steps, step_prefix,
                     gen_base, labels_dict, out_tag=""):
    """Plot per-leg per-step efficiency overlays for one chain type.

    """
    for L in range(n_legs):
        for var in ctx.vars:
            objs = []
            for S in steps:
                nm = f"eff_{particle}_dm{dm}_leg{L}_{step_prefix}{S}_{var}"
                obj = d_dm.Get(nm)
                if obj:
                    objs.append(obj)
                else:
                    ctx.missing.append(f"{ctx.dqm_dir}/GenDM{dm}/{nm}")
            if not objs:
                continue
            gen_obj = d_dm.Get(f"{gen_base}_dm{dm}_{var}")
            suffix = f"_{out_tag}" if out_tag else ""
            overlay_efficiency_with_gen(
                objs, gen_obj,
                os.path.join(dm_dir, f"eff_{particle}_dm{dm}_leg{L}{suffix}_{var}.png"),
                step_labels=labels_dict,
            )


def _collect_hists(d_dm, pattern_fn, items):
    """Return non-None histograms: [d_dm.Get(pattern_fn(s)) for s in items if exists]."""
    return list(filter(None, [d_dm.Get(pattern_fn(s)) for s in items]))

def plot_per_leg_efficiencies(ctx):
    for dm in ctx.dm_list:
        if dm not in ch_legs_by_dm:
            print(f"Skipping non-physical DM {dm}")
            continue
        d_dm = ctx.get_dm_tdir(dm)
        dm_dir = ctx.dm_out_dir(dm)

        # Calo chain: charged and photon
        _plot_chain_effs(ctx, d_dm, dm, dm_dir,
                         "ch", ch_legs_by_dm[dm], PLOT_STEPS, "step",
                         "cp_chHad", step_label_dict)
        _plot_chain_effs(ctx, d_dm, dm, dm_dir,
                         "pho", pho_legs_by_dm[dm], PLOT_STEPS, "step",
                         "cp_gamma", step_label_dict)

        # Track chain: charged only
        _plot_chain_effs(ctx, d_dm, dm, dm_dir,
                         "ch", ch_legs_by_dm[dm], PLOT_TRACK_STEPS, "trkstep",
                         "cp_chHad", track_step_label_dict, "trkchain")

        # Combi (AND) chain: charged only
        _plot_chain_effs(ctx, d_dm, dm, dm_dir,
                         "ch", ch_legs_by_dm[dm], PLOT_COMBI_STEPS, "combistep",
                         "cp_chHad", combi_step_label_dict, "combichain")

        # Combined calo + track overlay
        for L in range(ch_legs_by_dm[dm]):
            for var in ctx.vars:
                calo_list = _collect_hists(
                    d_dm, lambda S: f"eff_ch_dm{dm}_leg{L}_step{S}_{var}", PLOT_STEPS)
                track_list = _collect_hists(
                    d_dm, lambda S: f"eff_ch_dm{dm}_leg{L}_trkstep{S}_{var}", PLOT_TRACK_STEPS)
                if calo_list or track_list:
                    gen_obj = d_dm.Get(f"cp_chHad_dm{dm}_{var}")
                    overlay_calo_and_track_chain(
                        calo_list, track_list, gen_obj,
                        os.path.join(dm_dir, f"eff_ch_dm{dm}_leg{L}_calo_vs_track_{var}.png"),
                    )


def plot_tau_level_efficiencies(ctx):
    """Section 2: Tau-level efficiencies vs mother tau kinematics."""
    eff_all_pt, eff_all_eta = [], []
    eff_ge_ch_pt, eff_ge_ch_eta = {}, {}
    eff_ge_pi0_pt, eff_ge_pi0_eta = {}, {}
    special_all_map = {0: 1, 10: 3}

    for dm in ctx.dm_list:
        if dm not in ch_legs_by_dm:
            continue
        d_dm = ctx.get_dm_tdir(dm)
        dm_dir = ctx.dm_out_dir(dm)
        dm_path = f"{ctx.dqm_dir}/GenDM{dm}"

        # >= N charged
        for N in range(1, ch_legs_by_dm[dm] + 1):
            for var in ctx.vars:
                nm = f"eff_tau_dm{dm}_ge{N}ch_{var}"
                obj = d_dm.Get(nm)
                if obj:
                    if var == "pt":
                        eff_ge_ch_pt.setdefault(N, []).append(obj)
                    elif var == "eta":
                        eff_ge_ch_eta.setdefault(N, []).append(obj)
                else:
                    ctx.missing.append(f"{dm_path}/{nm}")

        # >= N pi0
        for N in range(1, pi0_cap_by_dm[dm] + 1):
            for var in ctx.vars:
                nm = f"eff_tau_dm{dm}_ge{N}pi0_{var}"
                obj = d_dm.Get(nm)
                if obj:
                    if var == "pt":
                        eff_ge_pi0_pt.setdefault(N, []).append(obj)
                    elif var == "eta":
                        eff_ge_pi0_eta.setdefault(N, []).append(obj)
                else:
                    ctx.missing.append(f"{dm_path}/{nm}")

        # ALL expected
        for var in ctx.vars:
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
                    ctx.missing.append(f"{dm_path}/{alt_nm}")

            if not obj and dm not in special_all_map and not overlay_obj:
                ctx.missing.append(f"{dm_path}/{nm}")

            if overlay_obj:
                if var == "pt":
                    eff_all_pt.append((f"DM {dm}", overlay_obj))
                elif var == "eta":
                    eff_all_eta.append((f"DM {dm}", overlay_obj))

        # Signal / iso split — overlay signal vs iso per DM
        for N in range(1, ch_legs_by_dm[dm] + 1):
            for var in ctx.vars:
                objs = []
                for kind in ["signal", "iso"]:
                    nm = f"eff_tau_dm{dm}_ge{N}ch_{kind}_{var}"
                    obj = d_dm.Get(nm)
                    if obj:
                        obj.SetTitle(kind.capitalize())
                        objs.append(obj)
                    else:
                        ctx.missing.append(f"{dm_path}/{nm}")
                if objs:
                    overlay_efficiency(objs, os.path.join(dm_dir, f"eff_tau_dm{dm}_ge{N}ch_signal_iso_{var}.png"))

        for N in range(1, pi0_cap_by_dm[dm] + 1):
            for var in ctx.vars:
                objs = []
                for kind in ["signal", "iso"]:
                    nm = f"eff_tau_dm{dm}_ge{N}pi0_{kind}_{var}"
                    obj = d_dm.Get(nm)
                    if obj:
                        obj.SetTitle(kind.capitalize())
                        objs.append(obj)
                    else:
                        ctx.missing.append(f"{dm_path}/{nm}")
                if objs:
                    overlay_efficiency(objs, os.path.join(dm_dir, f"eff_tau_dm{dm}_ge{N}pi0_signal_iso_{var}.png"))

        # Two-fold: overlay track / calo / both / either per ≥N charged
        for N in range(1, ch_legs_by_dm[dm] + 1):
            for var in ctx.vars:
                objs = []
                for kind, label in [("track", "Track"), ("calo", "Calo"),
                                    ("both", "Track AND Calo"), ("either", "Track OR Calo")]:
                    nm = f"eff_tau_dm{dm}_ge{N}ch_{kind}_{var}"
                    obj = d_dm.Get(nm)
                    if obj:
                        obj.SetTitle(label)
                        objs.append(obj)
                    else:
                        ctx.missing.append(f"{dm_path}/{nm}")
                if objs:
                    overlay_efficiency(objs, os.path.join(dm_dir, f"eff_tau_dm{dm}_ge{N}ch_twofold_{var}.png"))

    # Cross-DM overlays
    if eff_all_pt:
        _, hists = zip(*eff_all_pt)
        overlay_efficiency(list(hists), os.path.join(ctx.out_dir, "tau_eff_all_dm_overlay_pt.png"))

    if PLOT_ETA and eff_all_eta:
        _, hists = zip(*eff_all_eta)
        overlay_efficiency(list(hists), os.path.join(ctx.out_dir, "tau_eff_all_dm_overlay_eta.png"))

    for N, hists in sorted(eff_ge_ch_pt.items()):
        overlay_efficiency(list(hists), os.path.join(ctx.out_dir, f"tau_eff_ge{N}ch_dm_overlay_pt.png"))

    if PLOT_ETA:
        for N, hists in sorted(eff_ge_ch_eta.items()):
            overlay_efficiency(list(hists), os.path.join(ctx.out_dir, f"tau_eff_ge{N}ch_dm_overlay_eta.png"))

    for N, hists in sorted(eff_ge_pi0_pt.items()):
        overlay_efficiency(list(hists), os.path.join(ctx.out_dir, f"tau_eff_ge{N}pi0_dm_overlay_pt.png"))

    if PLOT_ETA:
        for N, hists in sorted(eff_ge_pi0_eta.items()):
            overlay_efficiency(list(hists), os.path.join(ctx.out_dir, f"tau_eff_ge{N}pi0_dm_overlay_eta.png"))

    # Cross-DM overlays for two-fold kinds (at the "all charged" level per DM)
    for kind in ["track", "calo", "both", "either"]:
        for var in ctx.vars:
            objs = []
            for dm in ctx.dm_list:
                if dm not in ch_legs_by_dm:
                    continue
                d_dm = ctx.get_dm_tdir(dm)
                if not d_dm:
                    continue
                N_all = ch_legs_by_dm[dm]
                nm = f"eff_tau_dm{dm}_ge{N_all}ch_{kind}_{var}"
                obj = d_dm.Get(nm)
                if obj:
                    obj.SetTitle(f"DM {dm}")
                    objs.append(obj)
            if objs:
                overlay_efficiency(objs, os.path.join(ctx.out_dir, f"tau_eff_allCh_{kind}_dm_overlay_{var}.png"))


def plot_inputs_and_matrices(ctx):
    """ Global inputs and confusion matrices."""
    d = ctx.base_tdir

    reco_labels = [
        "h^{#pm}", "h^{#pm}#pi^{0}", "h^{#pm}#pi^{0}#pi^{0}",
        "h^{#pm}h^{#pm}", "h^{#pm}h^{#pm}h^{#pm}", "h^{#pm}h^{#pm}h^{#pm}#pi^{0}",
    ]
    gen_labels = [
        "h^{#pm}", "h^{#pm}#pi^{0}", "h^{#pm}#pi^{0}#pi^{0}",
        "h^{#pm}h^{#pm}h^{#pm}", "h^{#pm}h^{#pm}h^{#pm}#pi^{0}",
    ]

    confusion_names = {"dm_reco_vs_gen_jet", "dm_reco_vs_gen_tau", "dm_reco_vs_gen_hps"}
    confusion_xtitles = {
        "dm_reco_vs_gen_jet": "Reco DM in jet",
        "dm_reco_vs_gen_tau": "Reco DM in tau",
        "dm_reco_vs_gen_hps": "DM assigned by HPS",
    }

    global_names = [
        "cp_chHad_pt_all", "cp_chHad_eta_all",
        "cp_gamma_pt_all", "cp_gamma_eta_all",
    ] + list(confusion_names)

    for name in global_names:
        obj = d.Get(name)
        if not obj:
            ctx.missing.append(f"{ctx.dqm_dir}/{name}")
            continue
        out = os.path.join(ctx.out_dir, f"{name}.png")
        if obj.InheritsFrom("TH2"):
            if name in confusion_names:
                xax, yax = obj.GetXaxis(), obj.GetYaxis()
                for i, lab in enumerate(reco_labels, start=1):
                    if i <= xax.GetNbins():
                        xax.SetBinLabel(i, lab)
                for j, lab in enumerate(gen_labels, start=1):
                    if j <= yax.GetNbins():
                        yax.SetBinLabel(j, lab)
                xax.SetTitle(confusion_xtitles[name])
                yax.SetTitle("Gen DM")
            draw_hist(obj, out, "COLZ")
        else:
            draw_hist(obj, out, "HIST E1")


def plot_tau_raw_inputs(ctx):
    """Full tau-level histograms (raw inputs) and DM overlays."""
    gen_pt, reco_pt = [], []
    gen_eta, reco_eta = [], []

    for dm in ctx.dm_list:
        if dm not in ch_legs_by_dm:
            print(f"Skipping non-physical DM {dm}")
            continue
        d_dm = ctx.get_dm_tdir(dm)
        if not d_dm:
            ctx.missing.append(f"{ctx.dqm_dir}/GenDM{dm}")
            continue
        dm_dir = ctx.dm_out_dir(dm)
        dm_path = f"{ctx.dqm_dir}/GenDM{dm}"

        # Raw histogram groups: (name pattern, count range)
        raw_groups = [
            (lambda v: f"tau_dm{dm}_den_{v}", 1),
            (lambda v: f"tau_dm{dm}_all_num_{v}", 1),
            (lambda v: f"tau_dm{dm}_reco_{v}", 1),
        ]
        for N in range(1, ch_legs_by_dm[dm] + 1):
            raw_groups.append((lambda v, n=N: f"tau_dm{dm}_ge{n}ch_num_{v}", 1))
        for N in range(1, pi0_cap_by_dm[dm] + 1):
            raw_groups.append((lambda v, n=N: f"tau_dm{dm}_ge{n}pi0_num_{v}", 1))

        for name_fn, _ in raw_groups:
            for var in ctx.vars:
                nm = name_fn(var)
                obj = ctx.get_or_miss(d_dm, nm, dm_path)
                if obj:
                    draw_hist(obj, os.path.join(dm_dir, f"{nm}.png"), "HIST E1")

        # Collect for cross-DM overlays
        h = ctx.get_or_miss(d_dm, f"tau_dm{dm}_den_pt", dm_path)
        if h:
            gen_pt.append((f"DM {dm}", h))
        h = ctx.get_or_miss(d_dm, f"tau_dm{dm}_reco_pt", dm_path)
        if h:
            reco_pt.append((f"DM {dm}", h))
        if PLOT_ETA:
            h = ctx.get_or_miss(d_dm, f"tau_dm{dm}_den_eta", dm_path)
            if h:
                gen_eta.append((f"DM {dm}", h))
            h = ctx.get_or_miss(d_dm, f"tau_dm{dm}_reco_eta", dm_path)
            if h:
                reco_eta.append((f"DM {dm}", h))

    # Cross-DM overlays
    overlay_specs = [
        (gen_pt,   "tau_gen_pt_dm_overlay.png",   r"$p_{T}(\tau)$ [GeV]", "Gen $\\tau$ $p_{T}$",   True),
        (reco_pt,  "tau_reco_pt_dm_overlay.png",  r"$p_{T}(\tau)$ [GeV]", "Reco $\\tau$ $p_{T}$",  True),
        (gen_eta,  "tau_gen_eta_dm_overlay.png",   r"$\eta(\tau)$",        "Gen $\\tau$ $\\eta$",    PLOT_ETA),
        (reco_eta, "tau_reco_eta_dm_overlay.png",  r"$\eta(\tau)$",        "Reco $\\tau$ $\\eta$",   PLOT_ETA),
    ]
    for hist_list, fname, xlabel, title, enabled in overlay_specs:
        if enabled and hist_list:
            labels, hists = zip(*hist_list)
            overlay_hist(list(hists), list(labels),
                         os.path.join(ctx.out_dir, fname), xlabel, "Entries", title)


def plot_pt_resolution(ctx):
    """ pT resolution plots (pt_reco / pt_sim)."""
    res_hists = []
    for dm in ctx.dm_list:
        d_dm = ctx.get_dm_tdir(dm)
        if not d_dm:
            ctx.missing.append(f"{ctx.dqm_dir}/GenDM{dm}")
            continue
        hname = f"tau_dm{dm}_pt_reco_over_gen"
        h = ctx.get_or_miss(d_dm, hname, f"{ctx.dqm_dir}/GenDM{dm}")
        if not h:
            continue
        res_hists.append((dm, h))

    if res_hists:
        objs = [h for _, h in res_hists]
        labels = [f"DM {dm}" for dm, _ in res_hists]
        overlay_hist(objs, labels,
                     os.path.join(ctx.out_dir, "tau_pt_reco_over_gen_overlay.png"),
                     r"$p_{T}^{reco}(\tau) / p_{T}^{gen}(\tau)$", "Entries",
                     "$\\tau$ $p_{T}$ resolution")


def plot_cp_base_histograms(ctx):
    """ Per-DM CP base histograms."""
    for dm in ctx.dm_list:
        d_dm = ctx.get_dm_tdir(dm)
        if not d_dm:
            ctx.missing.append(f"{ctx.dqm_dir}/GenDM{dm}")
            continue
        dm_dir = ctx.dm_out_dir(dm)
        for nm in [f"cp_chHad_dm{dm}_pt", f"cp_chHad_dm{dm}_eta",
                    f"cp_gamma_dm{dm}_pt", f"cp_gamma_dm{dm}_eta"]:
            obj = d_dm.Get(nm)
            if obj:
                draw_hist(obj, os.path.join(dm_dir, f"{nm}.png"), "HIST E1")
            else:
                ctx.missing.append(f"{ctx.dqm_dir}/GenDM{dm}/{nm}")


def plot_cp_pf_resolution(ctx):
    """ CP-to-PF pT resolution (1D ratio histograms, per DM)."""
    cp_pf_had, cp_pf_em = [], []

    for dm in ctx.dm_list:
        d_dm = ctx.get_dm_tdir(dm)
        if not d_dm:
            ctx.missing.append(f"{ctx.dqm_dir}/GenDM{dm}")
            continue
        dm_path = f"{ctx.dqm_dir}/GenDM{dm}"

        for suffix, accum in [("hadronic", cp_pf_had), ("em", cp_pf_em)]:
            nm = f"cp_pf_pt_resolution_{suffix}_dm{dm}"
            obj = d_dm.Get(nm)
            if obj:
                accum.append((dm, obj))
            else:
                ctx.missing.append(f"{dm_path}/{nm}")

    if cp_pf_had:
        objs = [h for _, h in cp_pf_had]
        labels = [f'DM {dm}' for dm, _ in cp_pf_had]
        overlay_hist(objs, labels,
                     os.path.join(ctx.out_dir, "cp_pf_pt_resolution_hadronic_overlay.png"),
                     r"$p_{T}^{reco}(\pi^{\pm})/p_{T}^{gen}(\pi^{\pm})$", "Entries",
                     "Hadronic CP resolution")
    if cp_pf_em:
        objs = [h for _, h in cp_pf_em]
        labels = [f'DM {dm}' for dm, _ in cp_pf_em]
        overlay_hist(objs, labels,
                     os.path.join(ctx.out_dir, "cp_pf_pt_resolution_em_overlay.png"),
                     r"$p_{T}^{reco}(\gamma)/p_{T}^{gen}(\gamma)$", "Entries",
                     "Gamma CP resolution")


def plot_fake_rates(ctx):
    """ Fake rate plots."""
    d_fake = ctx.file.Get(ctx.dqm_dir + "/FakeRate")
    if not d_fake:
        print(f"WARNING: FakeRate directory not found at {ctx.dqm_dir}/FakeRate")
        return

    fake_dm_sel = [0, 1, 2, 5, 10, 11]

    # Helper: plot one fake rate set (inclusive + per-DM + raw inputs + overlays)
    def _plot_fake_set(prefix):
        """Plot fake rate histos for a given prefix (e.g. 'fake', 'fake_calo', 'fake_track').
        Inclusive histos live in FakeRate/, per-DM histos in GenDM{dm}/FakeRate/."""

        # Inclusive fake rate — standalone removed; shown in assoc overlays
        for var in ctx.vars:
            nm = f"{prefix}_rate_{var}"
            if not d_fake.Get(nm):
                ctx.missing.append(f"{ctx.dqm_dir}/FakeRate/{nm}")

        # Inclusive raw inputs (from FakeRate/)
        for var in ctx.vars:
            for kind in ["den", "num"]:
                nm = f"{prefix}_{kind}_{var}"
                obj = d_fake.Get(nm)
                if obj:
                    draw_hist(obj, os.path.join(ctx.out_dir, f"{nm}.png"), "HIST E1")
                else:
                    ctx.missing.append(f"{ctx.dqm_dir}/FakeRate/{nm}")

        # Per reco-DM fake rate (from GenDM{dm}/FakeRate/)
        dm_pt_hists, dm_eta_hists = [], []
        for dm in fake_dm_sel:
            d_dm_fake = ctx.file.Get(f"{ctx.dqm_dir}/GenDM{dm}/FakeRate")
            dm_out = ctx.dm_out_dir(dm)

            for var in ctx.vars:
                nm = f"{prefix}_rate_dm{dm}_{var}"
                obj = d_dm_fake.Get(nm) if d_dm_fake else None
                if obj:
                    if var == "pt":
                        dm_pt_hists.append(obj)
                    elif var == "eta":
                        dm_eta_hists.append(obj)
                else:
                    ctx.missing.append(f"{ctx.dqm_dir}/GenDM{dm}/FakeRate/{nm}")

            # Per-DM raw inputs
            for var in ctx.vars:
                for kind in ["den", "num"]:
                    nm = f"{prefix}_dm{dm}_{kind}_{var}"
                    obj = d_dm_fake.Get(nm) if d_dm_fake else None
                    if obj:
                        draw_hist(obj, os.path.join(dm_out, f"{nm}.png"), "HIST E1")
                    else:
                        ctx.missing.append(f"{ctx.dqm_dir}/GenDM{dm}/FakeRate/{nm}")

        if dm_pt_hists:
            overlay_efficiency(dm_pt_hists, os.path.join(ctx.out_dir, f"{prefix}_rate_dm_overlay_pt.png"))
        if PLOT_ETA and dm_eta_hists:
            overlay_efficiency(dm_eta_hists, os.path.join(ctx.out_dir, f"{prefix}_rate_dm_overlay_eta.png"))

    # Combined (calo OR track)
    _plot_fake_set("fake")

    # Calo-only association
    _plot_fake_set("fake_calo")

    # Track-only association
    _plot_fake_set("fake_track")

    # Charged iso path filtered taus
    _plot_fake_set("fake_chargedIsoPath")
    _plot_fake_set("fake_calo_chargedIsoPath")
    _plot_fake_set("fake_track_chargedIsoPath")

    # Overlay: combined vs calo vs track (inclusive)
    for var in ctx.vars:
        objs = []
        for prefix, label in [("fake", "Combined (calo OR track)"),
                               ("fake_calo", "Calo (TICL) only"),
                               ("fake_track", "Track only")]:
            nm = f"{prefix}_rate_{var}"
            obj = d_fake.Get(nm)
            if obj:
                obj.SetTitle(label)
                objs.append(obj)
        if objs:
            overlay_efficiency(objs, os.path.join(ctx.out_dir, f"fake_rate_assoc_overlay_{var}.png"))

    # Overlay: combined vs calo vs track (per-DM)
    for dm in fake_dm_sel:
        d_dm_fake = ctx.file.Get(f"{ctx.dqm_dir}/GenDM{dm}/FakeRate")
        for var in ctx.vars:
            objs = []
            for prefix, label in [("fake", "Combined"),
                                   ("fake_calo", "Calo only"),
                                   ("fake_track", "Track only")]:
                nm = f"{prefix}_rate_dm{dm}_{var}"
                obj = d_dm_fake.Get(nm) if d_dm_fake else None
                if obj:
                    obj.SetTitle(f"{label}: DM {dm}")
                    objs.append(obj)
            if objs:
                overlay_efficiency(objs, os.path.join(ctx.dm_out_dir(dm), f"fake_rate_dm{dm}_assoc_overlay_{var}.png"))

    # Overlay: filtered combined vs calo vs track
    for var in ctx.vars:
        objs = []
        for prefix, label in [("fake_chargedIsoPath", "Combined (calo OR track)"),
                               ("fake_calo_chargedIsoPath", "Calo (TICL) only"),
                               ("fake_track_chargedIsoPath", "Track only")]:
            nm = f"{prefix}_rate_{var}"
            obj = d_fake.Get(nm)
            if obj:
                obj.SetTitle(label)
                objs.append(obj)
        if objs:
            overlay_efficiency(objs, os.path.join(ctx.out_dir, f"fake_rate_chargedIsoPath_assoc_overlay_{var}.png"))

    # Overlay: unfiltered vs HLT-filtered per association type (inclusive)
    for assoc, tag in [("fake", "Combined"), ("fake_calo", "Calo"), ("fake_track", "Track")]:
        filt = f"{assoc}_chargedIsoPath" if assoc == "fake" else f"{assoc}_chargedIsoPath"
        for var in ctx.vars:
            objs = []
            for prefix, label in [(assoc, f"{tag} (all reco taus)"),
                                   (filt, f"{tag} (HLT filtered)")]:
                nm = f"{prefix}_rate_{var}"
                obj = d_fake.Get(nm)
                if obj:
                    obj.SetTitle(label)
                    objs.append(obj)
            if objs:
                overlay_efficiency(objs, os.path.join(ctx.out_dir, f"{assoc}_rate_unfilt_vs_filt_{var}.png"))

    # Overlay: unfiltered vs HLT-filtered per association type (per-DM)
    for dm in fake_dm_sel:
        d_dm_fake = ctx.file.Get(f"{ctx.dqm_dir}/GenDM{dm}/FakeRate")
        if not d_dm_fake:
            continue
        for assoc, tag in [("fake", "Combined"), ("fake_calo", "Calo"), ("fake_track", "Track")]:
            filt = f"{assoc}_chargedIsoPath" if assoc == "fake" else f"{assoc}_chargedIsoPath"
            for var in ctx.vars:
                objs = []
                for prefix, label in [(assoc, f"{tag} (all)"),
                                       (filt, f"{tag} (HLT filtered)")]:
                    nm = f"{prefix}_rate_dm{dm}_{var}"
                    obj = d_dm_fake.Get(nm)
                    if obj:
                        obj.SetTitle(label)
                        objs.append(obj)
                if objs:
                    overlay_efficiency(objs, os.path.join(ctx.dm_out_dir(dm),
                                      f"{assoc}_rate_dm{dm}_unfilt_vs_filt_{var}.png"))

def main():
    global args
    parser = argparse.ArgumentParser(description='Make Ticl Tau validation plots.')
    parser.add_argument('-s', '--step', type=str, default='HLT',
                        help='Validation step ("HLT" or "Offline")')
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='Paths to the DQM ROOT file.')
    parser.add_argument('-o', '--odir', type=str, default="TauValidationPlots", required=False,
                        help='Path to the output directory.')
    parser.add_argument('-l', '--sample_label', type=str, default="Tau (200 PU)", required=False,
                        help='Sample label for plotting.')
    args = parser.parse_args()

    if args.step == 'HLT':
        dqm_dir = "DQMData/Run 1/HLT/Run summary/TICL/ticlTauValidator"
    elif args.step == 'Offline':
        dqm_dir = "DQMData/Run 1/Run summary/RecoTauV/ticlTauValidator/"
    else:
        sys.exit("### ERROR: Please chose the step among the following ['HLT', 'Offline']")

    root_file = ROOT.TFile.Open(args.file)
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Failed to open DQM file: {args.file}")
    if not root_file.Get(dqm_dir):
        raise RuntimeError(f"Directory '{dqm_dir}' not found in {args.file}")

    os.makedirs(args.odir, exist_ok=True)
    ctx = PlotContext(root_file, dqm_dir, args.odir)
    print(ctx.dm_subdirs)

    plot_per_leg_efficiencies(ctx)
    plot_tau_level_efficiencies(ctx)
    plot_inputs_and_matrices(ctx)
    plot_tau_raw_inputs(ctx)
    plot_pt_resolution(ctx)
    plot_cp_base_histograms(ctx)
    plot_cp_pf_resolution(ctx)
    plot_fake_rates(ctx)

    if ctx.missing:
        print("Missing objects:")
        for m in sorted(set(ctx.missing)):
            print("  -", m)

    root_file.Close()
    print(f"Done. Plots saved to: {ctx.out_dir}")


if __name__ == '__main__':
    main()
