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

# Truth photon CP cap per generated decay mode
pho_legs_by_dm = {0:0, 1:2, 2:4, 10:0, 11:2}

# Truth pi0-equivalent cap for tau-level summaries
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
    0: "Track step 0: CP→SimCand",
    1: "Track step 1: SimCand→Track",
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

def histo_values_2D(h):
    Nx = h.GetNbinsX()
    Ny = h.GetNbinsY()
    return np.array([
        [h.GetBinContent(i+1, j+1) for i in range(Nx)]
        for j in range(Ny)
    ])

def histo_values_errors(h):
    N = h.GetNbinsX()
    values = np.array([h.GetBinContent(i+1) for i in range(N)])
    errors = np.array([h.GetBinError(i+1) for i in range(N)])
    return values, errors

def overlay_efficiency(list_objs, out, legend_title_override=None, xlabel=None):

    fontsize = 20
    fig, ax = plt.subplots(figsize=(10, 10))
    hep.cms.label(llabel='Simulation Preliminary', rlabel=args.sample_label, ax=ax, fontsize=fontsize)

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
    
    def _endpoint_region(names):
        """Infer endpoint region from histogram names."""
        if any("_track_" in h for h in names):
            return "track"
        if any("_calo_" in h for h in names):
            return "calo"
        if any("_both_" in h for h in names):
            return "both"
        if any("_either_" in h for h in names):
            return "either"
        if any("_signal_" in h for h in names):
            return "signal"
        if any("_iso_" in h for h in names):
            return "iso"
        return "endpoint"

    mode_label = {
        "track": " (track-matched)",
        "calo": " (TICL-matched)",
        "both": " (track AND TICL)",
        "either": " (track OR TICL)",
        "signal": " (signal region)",
        "iso": " (isolation region)",
        "endpoint": " (tau endpoint)",
    }

    if legend_title_override is None:
        if has_all:
            legend_title = "Full tau efficiency"
        elif has_ge_ch:
            # Extract N from first _geNch histogram
            for h in hnames:
                if '_ge' in h and 'ch_' in h:
                    match = re.search(r'_ge(\d+)ch', h)
                    if match:
                        N = match.group(1)
                        mode = _endpoint_region(hnames)
                        legend_title = f"≥{N} charged hadron efficiency{mode_label.get(mode, '')}"
                        break
        elif has_ge_pi0:
            # Extract N from first _geNpi0 histogram
            for h in hnames:
                if '_ge' in h and 'pi0_' in h:
                    match = re.search(r'_ge(\d+)pi0', h)
                    if match:
                        N = match.group(1)
                        mode = _endpoint_region(hnames)
                        legend_title = f"≥{N} truth π⁰-equivalent efficiency{mode_label.get(mode, '')}"
                        break
    else:
        legend_title = legend_title_override
                            
    # Extract variable from histogram name (pt or eta)
    hname = list_objs[0].GetName()
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    elif '_pt' in hname:
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
    hep.cms.label(llabel='Simulation Preliminary', rlabel=args.sample_label, ax=ax1, fontsize=fontsize)
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
    hep.cms.label(llabel='Simulation Preliminary', rlabel=args.sample_label, ax=ax1, fontsize=fontsize)

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
    if args.norm_to_step0:
        chain_title += "\nEff. relative to step 0"
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

def plot_confusion_matrix(h2, out, xlabel, xticklabels, yticklabels):
    """Draw a 2D confusion matrix with the CMS label and per-cell counts."""
    fontsize = 20
    Nx, Ny, x_edges, y_edges = define_bins_2D(h2)
    values = histo_values_2D(h2)
    fig, ax = plt.subplots(figsize=(11, 9))
    hep.cms.label(llabel='Simulation Preliminary', rlabel=args.sample_label, ax=ax, fontsize=fontsize)
    mesh = ax.pcolormesh(x_edges, y_edges, values, cmap='viridis')
    fig.colorbar(mesh, ax=ax, label='Entries')
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    ax.set_xticks(xc[:len(xticklabels)])
    ax.set_xticklabels(xticklabels, fontsize=fontsize - 6)
    ax.set_yticks(yc[:len(yticklabels)])
    ax.set_yticklabels(yticklabels, fontsize=fontsize - 6)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel("Gen DM", fontsize=fontsize)
    vmax = values.max() if values.size else 0.
    if vmax > 0:
        for j in range(Ny):
            for i in range(Nx):
                v = values[j, i]
                if v > 0:
                    ax.text(xc[i], yc[j], f"{v:.0f}", ha='center', va='center',
                            color='black' if v > 0.6 * vmax else 'white', fontsize=fontsize - 6)
    plt.tight_layout()
    print(f'Saving plot: {out}')
    plt.savefig(out)
    plt.close()

def overlay_hist(list_objs, labels, out, xlabel, ylabel, title=""):

    fontsize = 20
    fig, ax = plt.subplots(figsize=(10, 10))
    hep.cms.label(llabel='Simulation Preliminary', rlabel=args.sample_label, ax=ax, fontsize=fontsize)

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
        # GenDM5 exists only for the FakeRate subdir (reco 2-prong category); keep gen DMs only
        self.dm_list = [int(s.replace("GenDM", "")) for s in self.dm_subdirs]
        self.dm_list = [dm for dm in self.dm_list if dm in ch_legs_by_dm]
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


def _make_conditional_eff(d_dm, particle, dm, L, step_prefix, S, var):
    """Efficiency of step S conditional on step 0: num(stepS)/num(step0), binomial errors.
    Valid because the step numerators are nested (pass(S) implies pass(0))."""
    num = d_dm.Get(f"{particle}_dm{dm}_leg{L}_{step_prefix}{S}_num_{var}")
    den = d_dm.Get(f"{particle}_dm{dm}_leg{L}_{step_prefix}0_num_{var}")
    if not num or not den:
        return None
    h = num.Clone(f"eff_{particle}_dm{dm}_leg{L}_{step_prefix}{S}_{var}_cond")
    h.SetDirectory(0)
    h.Divide(num, den, 1.0, 1.0, "B")
    vs = 'pT' if var == 'pt' else 'eta'
    h.SetTitle(f"DM {dm} {particle} leg{L} {step_prefix}{S}: efficiency (cond. on step0) vs {vs}")
    return h


def _get_step_eff(ctx, d_dm, particle, dm, L, step_prefix, S, var):
    """Fetch the harvested efficiency, or the step0-conditional one if requested.
    Conditional normalization only applies to the calo ('step') and track ('trkstep') chains."""
    if args.norm_to_step0 and S > 0 and step_prefix in ("step", "trkstep"):
        obj = _make_conditional_eff(d_dm, particle, dm, L, step_prefix, S, var)
        if obj:
            return obj
    nm = f"eff_{particle}_dm{dm}_leg{L}_{step_prefix}{S}_{var}"
    obj = d_dm.Get(nm)
    if not obj:
        ctx.missing.append(f"{ctx.dqm_dir}/GenDM{dm}/{nm}")
    return obj


def _plot_chain_effs(ctx, d_dm, dm, dm_dir,
                     particle, n_legs, steps, step_prefix,
                     gen_base, labels_dict, out_tag=""):
    """Plot per-leg per-step efficiency overlays for one chain type.

    """
    for L in range(n_legs):
        for var in ctx.vars:
            objs = []
            for S in steps:
                obj = _get_step_eff(ctx, d_dm, particle, dm, L, step_prefix, S, var)
                if obj:
                    objs.append(obj)
            if not objs:
                continue
            gen_obj = d_dm.Get(f"{gen_base}_dm{dm}_{var}")
            suffix = f"_{out_tag}" if out_tag else ""
            if args.norm_to_step0 and step_prefix in ("step", "trkstep"):
                suffix += "_cond"
            overlay_efficiency_with_gen(
                objs, gen_obj,
                os.path.join(dm_dir, f"eff_{particle}_dm{dm}_leg{L}{suffix}_{var}.png"),
                step_labels=labels_dict,
            )


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
                calo_list = list(filter(None, [
                    _get_step_eff(ctx, d_dm, "ch", dm, L, "step", S, var) for S in PLOT_STEPS]))
                track_list = list(filter(None, [
                    _get_step_eff(ctx, d_dm, "ch", dm, L, "trkstep", S, var) for S in PLOT_TRACK_STEPS]))
                if calo_list or track_list:
                    gen_obj = d_dm.Get(f"cp_chHad_dm{dm}_{var}")
                    cond_tag = "_cond" if args.norm_to_step0 else ""
                    overlay_calo_and_track_chain(
                        calo_list, track_list, gen_obj,
                        os.path.join(dm_dir, f"eff_ch_dm{dm}_leg{L}_calo_vs_track{cond_tag}_{var}.png"),
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

        # >= N truth pi0 equivalents
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

    # Cross-DM overlays for two-fold kinds:
    # - ge1ch: common >=1 charged threshold across DMs
    # - allCh: DM-dependent all-expected charged threshold (DM0/1/2: >=1, DM10/11: >=3)
    mode_label = {
        "track": " (track-matched)",
        "calo": " (TICL-matched)",
        "both": " (track AND TICL)",
        "either": " (track OR TICL)",
    }
    for kind in ["track", "calo", "both", "either"]:
        for var in ctx.vars:
            objs_ge1 = []
            objs_all = []
            for dm in ctx.dm_list:
                if dm not in ch_legs_by_dm:
                    continue
                d_dm = ctx.get_dm_tdir(dm)
                if not d_dm:
                    continue
                nm_ge1 = f"eff_tau_dm{dm}_ge1ch_{kind}_{var}"
                obj_ge1 = d_dm.Get(nm_ge1)
                if obj_ge1:
                    obj_ge1.SetTitle(f"DM {dm}")
                    objs_ge1.append(obj_ge1)

                n_all = ch_legs_by_dm[dm]
                nm_all = f"eff_tau_dm{dm}_ge{n_all}ch_{kind}_{var}"
                obj_all = d_dm.Get(nm_all)
                if obj_all:
                    obj_all.SetTitle(f"DM {dm}")
                    objs_all.append(obj_all)

            if objs_ge1:
                overlay_efficiency(
                    objs_ge1,
                    os.path.join(ctx.out_dir, f"tau_eff_ge1ch_{kind}_dm_overlay_{var}.png"),
                    legend_title_override=f"≥1 charged hadron efficiency{mode_label[kind]}",
                )
            if objs_all:
                overlay_efficiency(
                    objs_all,
                    os.path.join(ctx.out_dir, f"tau_eff_allCh_{kind}_dm_overlay_{var}.png"),
                    legend_title_override=f"All charged hadron efficiency{mode_label[kind]}",
                )


def plot_confusion_matrices(ctx):
    """DM confusion matrices (reco vs gen)."""
    d = ctx.base_tdir

    reco_labels = [
        r"$h^{\pm}$", r"$h^{\pm}\pi^{0}$", r"$h^{\pm}\pi^{0}\pi^{0}$",
        r"$h^{\pm}h^{\pm}$", r"$h^{\pm}h^{\pm}h^{\pm}$", r"$h^{\pm}h^{\pm}h^{\pm}\pi^{0}$",
    ]
    gen_labels = [
        r"$h^{\pm}$", r"$h^{\pm}\pi^{0}$", r"$h^{\pm}\pi^{0}\pi^{0}$",
        r"$h^{\pm}h^{\pm}h^{\pm}$", r"$h^{\pm}h^{\pm}h^{\pm}\pi^{0}$",
    ]
    confusion_xtitles = {
        "dm_reco_vs_gen_jet": "Truth CP coverage DM at PFJet",
        "dm_reco_vs_gen_tau": "Truth CP coverage DM in selected PFTau",
        "dm_reco_vs_gen_hps": "DM assigned by configured PFTau producer",
    }

    for name, xlabel in confusion_xtitles.items():
        obj = d.Get(name)
        if not obj:
            ctx.missing.append(f"{ctx.dqm_dir}/{name}")
            continue
        plot_confusion_matrix(obj, os.path.join(ctx.out_dir, f"{name}.png"),
                              xlabel, reco_labels, gen_labels)


def plot_tau_distributions(ctx):
    """Cross-DM overlays of gen and reco tau kinematics."""
    gen_pt, reco_pt = [], []
    gen_eta, reco_eta = [], []

    for dm in ctx.dm_list:
        d_dm = ctx.get_dm_tdir(dm)
        if not d_dm:
            ctx.missing.append(f"{ctx.dqm_dir}/GenDM{dm}")
            continue
        dm_path = f"{ctx.dqm_dir}/GenDM{dm}"

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


def plot_cp_twofold_efficiencies(ctx):
    """CP-level two-fold match efficiencies (track vs calo, harvested)."""
    cp_xlabel = {"pt": r"$p_{T}$(CP) [GeV]", "eta": r"$\eta$(CP)"}
    gamma_overlays = {var: [] for var in ctx.vars}

    for dm in ctx.dm_list:
        if dm not in ch_legs_by_dm:
            continue
        d_dm = ctx.get_dm_tdir(dm)
        if not d_dm:
            ctx.missing.append(f"{ctx.dqm_dir}/GenDM{dm}")
            continue
        dm_dir = ctx.dm_out_dir(dm)
        dm_path = f"{ctx.dqm_dir}/GenDM{dm}"

        for var in ctx.vars:
            # charged: overlay track / calo / both per DM
            objs = []
            for kind, label in [("trackOnly", "Track"), ("caloOnly", "Calo (TICL)"),
                                ("trackAndCalo", "Track AND Calo")]:
                nm = f"eff_cp_chHad_dm{dm}_{kind}_{var}"
                obj = d_dm.Get(nm)
                if obj:
                    obj.SetTitle(label)
                    objs.append(obj)
                else:
                    ctx.missing.append(f"{dm_path}/{nm}")
            if objs:
                overlay_efficiency(
                    objs,
                    os.path.join(dm_dir, f"eff_cp_chHad_dm{dm}_twofold_{var}.png"),
                    legend_title_override=f"Charged CP match efficiency (DM {dm})",
                    xlabel=cp_xlabel[var],
                )

            # photon: collect calo-only for cross-DM overlay (only DMs with photons)
            if pho_legs_by_dm.get(dm, 0) > 0:
                nm = f"eff_cp_gamma_dm{dm}_caloOnly_{var}"
                obj = d_dm.Get(nm)
                if obj:
                    obj.SetTitle(f"DM {dm}")
                    gamma_overlays[var].append(obj)
                else:
                    ctx.missing.append(f"{dm_path}/{nm}")

    for var, objs in gamma_overlays.items():
        if objs:
            overlay_efficiency(
                objs,
                os.path.join(ctx.out_dir, f"eff_cp_gamma_caloOnly_dm_overlay_{var}.png"),
                legend_title_override="Photon CP calo-match efficiency",
                xlabel=cp_xlabel[var],
            )


def plot_fake_rates(ctx):
    """ Fake rate plots."""
    d_fake = ctx.file.Get(ctx.dqm_dir + "/FakeRate")
    if not d_fake:
        print(f"WARNING: FakeRate directory not found at {ctx.dqm_dir}/FakeRate")
        return

    fake_dm_sel = [0, 1, 2, 5, 10, 11]

    # Helper: plot one fake rate set (per-DM rates + overlays)
    def _plot_fake_set(prefix):
        """Plot fake rate histos for a given prefix (e.g. 'fake', 'fake_calo', 'fake_track').
        Inclusive histos live in FakeRate/, per-DM histos in GenDM{dm}/FakeRate/."""

        # Inclusive fake rate — standalone removed; shown in assoc overlays
        for var in ctx.vars:
            nm = f"{prefix}_rate_{var}"
            if not d_fake.Get(nm):
                ctx.missing.append(f"{ctx.dqm_dir}/FakeRate/{nm}")

        # Per reco-DM fake rate (from GenDM{dm}/FakeRate/)
        dm_pt_hists, dm_eta_hists = [], []
        for dm in fake_dm_sel:
            d_dm_fake = ctx.file.Get(f"{ctx.dqm_dir}/GenDM{dm}/FakeRate")

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
    parser.add_argument('--norm-to-step0', dest='norm_to_step0', action='store_true',
                        help='Plot calo/track chain step efficiencies conditional on step 0 '
                             '(step-N numerator divided by step-0 numerator). '
                             'Step 0 itself stays unconditional (truth visibility).')
    args = parser.parse_args()

    module = "ticlTauValidator"

    if args.step == 'HLT':
        dqm_dir = f"DQMData/Run 1/HLT/Run summary/TICL/{module}"
    elif args.step == 'Offline':
        dqm_dir = f"DQMData/Run 1/Run summary/RecoTauV/{module}"
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
    plot_confusion_matrices(ctx)
    plot_tau_distributions(ctx)
    plot_pt_resolution(ctx)
    plot_cp_pf_resolution(ctx)
    plot_cp_twofold_efficiencies(ctx)
    plot_fake_rates(ctx)

    if ctx.missing:
        print("Missing objects:")
        for m in sorted(set(ctx.missing)):
            print("  -", m)

    root_file.Close()
    print(f"Done. Plots saved to: {ctx.out_dir}")


if __name__ == '__main__':
    main()
