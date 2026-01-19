import ROOT
import os
import glob
ROOT.gROOT.SetBatch(True)

# ======= CONFIG =======
PLOT_STEPS = [0, 1, 4, 5]
PLOT_ETA   = True              # enable eta plots

dm_list = [0, 1, 2, 5, 10, 11]

# Charged leg cap per DM (actual prongs)
ch_legs_by_dm = {0:1, 1:1, 2:1, 5:2, 10:3, 11:3}

# Photon leg cap per DM (only where pi0 exist); 2 photons per pi0
pho_legs_by_dm = {0:0, 1:2, 2:4, 5:0, 10:0, 11:4}

# Try current names first; fall back to older abs/cond names
def get_eff(d, nm_list):
    for nm in nm_list:
        obj = d.Get(nm)
        if obj: return obj, nm
    return None, None

def draw_eff(obj, out):
    c = ROOT.TCanvas("c","c",900,650); obj.SetStats(0)
    if obj.InheritsFrom("TEfficiency"):
        obj.Draw("AP"); ROOT.gPad.Update()
        g = obj.GetPaintedGraph()
        if g: g.GetYaxis().SetRangeUser(0.0,1.1); g.GetYaxis().SetTitle("Efficiency")
    elif obj.InheritsFrom("TH1"):
        obj.SetMinimum(0.0); obj.SetMaximum(1.1); obj.GetYaxis().SetTitle("Efficiency"); obj.Draw("E1")
    else:
        obj.Draw("AP"); ROOT.gPad.Update()
        try: obj.GetYaxis().SetRangeUser(0.0,1.1)
        except: pass
    c.SaveAs(out); c.Close()

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

# ---- Open file & locate folder ----
pattern = "/afs/cern.ch/work/a/agruber/private/HLT_Phase2/CMSSW_16_0_0_pre3/src/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root"
files = glob.glob(pattern)
if not files:
    raise RuntimeError(f"No files match pattern: {pattern}")
if len(files) > 1:
    print("Warning: multiple files found, using the first one:\n  ", files[0])

f = ROOT.TFile.Open(files[0])
if not f or f.IsZombie():
    raise RuntimeError(f"Failed to open harvested DQM file: {files[0]}")

# Hardcode the subdir here
subdir = "DQMData/Run 1/HLT/Run summary/TICL/ticlTauValidator"
d = f.Get(subdir)
if not d:
    raise RuntimeError(f"Directory not found: {subdir}")

# ---- Output dirs ----
out_dir = "plots"
eff_dir = os.path.join(out_dir, "efficiencies_trimmed")
mat_dir = os.path.join(out_dir, "matrices_inputs")
full_tau_dir = os.path.join(out_dir, "full_tau")  # new: raw tau-level inputs
os.makedirs(eff_dir, exist_ok=True)
os.makedirs(mat_dir, exist_ok=True)
os.makedirs(full_tau_dir, exist_ok=True)

missing = []

# ======= 1) Per-leg, per-step efficiencies (trimmed) =======
vars_to_plot = ["pt"] + (["eta"] if PLOT_ETA else [])
for dm in dm_list:
    # charged legs
    for L in range(ch_legs_by_dm[dm]):
        for S in PLOT_STEPS:
            for var in vars_to_plot:
                obj, used = get_eff(d, [
                    f"eff_ch_dm{dm}_leg{L}_step{S}_{var}",
                    f"eff_abs_ch_dm{dm}_leg{L}_step{S}_{var}",
                    f"eff_cond_ch_dm{dm}_leg{L}_step{S}_{var}",
                ])
                if obj: draw_eff(obj, os.path.join(eff_dir, f"{used}.png"))
                else:   missing.append(f"{subdir}/eff_ch_dm{dm}_leg{L}_step{S}_{var}")

    # photon legs (only if DM has pi0)
    for L in range(pho_legs_by_dm[dm]):
        for S in PLOT_STEPS:
            for var in vars_to_plot:
                obj, used = get_eff(d, [
                    f"eff_pho_dm{dm}_leg{L}_step{S}_{var}",
                    f"eff_abs_pho_dm{dm}_leg{L}_step{S}_{var}",
                    f"eff_cond_pho_dm{dm}_leg{L}_step{S}_{var}",
                ])
                if obj: draw_eff(obj, os.path.join(eff_dir, f"{used}.png"))
                else:   missing.append(f"{subdir}/eff_pho_dm{dm}_leg{L}_step{S}_{var}")

# ======= 2) Tau-level efficiencies vs mother τ kinematics =======
# (kept as-is; they’re already compact and per DM)
for dm in dm_list:
    # ≥N charged (N up to actual prongs)
    for N in range(1, ch_legs_by_dm[dm] + 1):
        for var in vars_to_plot:
            nm = f"eff_tau_dm{dm}_ge{N}ch_{var}"
            obj = d.Get(nm)
            if obj: draw_eff(obj, os.path.join(eff_dir, f"{nm}.png"))
            else:   missing.append(f"{subdir}/{nm}")
    # ≥N pi0 (N up to pi0 count per DM)
    pi0_cap = {0:0,1:1,2:2,5:0,10:0,11:1}[dm]  # DM11 has 1 pi0
    for N in range(1, pi0_cap + 1):
        for var in vars_to_plot:
            nm = f"eff_tau_dm{dm}_ge{N}pi0_{var}"
            obj = d.Get(nm)
            if obj: draw_eff(obj, os.path.join(eff_dir, f"{nm}.png"))
            else:   missing.append(f"{subdir}/{nm}")
    # ALL expected (now for all DMs, including 0 and 10)
    for var in vars_to_plot:
        nm = f"eff_tau_dm{dm}_all_{var}"
        obj = d.Get(nm)
        if obj: draw_eff(obj, os.path.join(eff_dir, f"{nm}.png"))
        else:   missing.append(f"{subdir}/{nm}")

    # TAU-only split efficiencies (signal / iso)
    for N in range(1, ch_legs_by_dm[dm] + 1):
        for var in vars_to_plot:
            for kind in ["signal", "iso"]:
                nm = f"eff_tau_dm{dm}_ge{N}ch_{kind}_{var}"
                obj = d.Get(nm)
                if obj: draw_eff(obj, os.path.join(eff_dir, f"{nm}.png"))
                else:   missing.append(f"{subdir}/{nm}")
    pi0_cap = {0:0,1:1,2:2,5:0,10:0,11:1}[dm]
    for N in range(1, pi0_cap + 1):
        for var in vars_to_plot:
            for kind in ["signal", "iso"]:
                nm = f"eff_tau_dm{dm}_ge{N}pi0_{kind}_{var}"
                obj = d.Get(nm)
                if obj: draw_eff(obj, os.path.join(eff_dir, f"{nm}.png"))
                else:   missing.append(f"{subdir}/{nm}")

# ======= 3) Inputs & matrices (updated: only new confusion matrices) =======
inputs_to_plot = [
    "cp_chHad_pt_all", "cp_chHad_eta_all",
    "cp_gamma_pt_all", "cp_gamma_eta_all",
    "dm_reco_vs_gen_jet",
    "dm_reco_vs_gen_tau",
    "dm_reco_vs_gen_hps",
]
for dm in dm_list:
    inputs_to_plot += [f"dm{dm}_tauRecoMatrix", f"dm{dm}_nRecoCh", f"dm{dm}_nRecoPi0"]

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

for name in inputs_to_plot:
    obj = d.Get(name)
    if not obj:
        missing.append(f"{subdir}/{name}")
        continue
    out = os.path.join(mat_dir, f"{name}.png")
    if obj.InheritsFrom("TH2"):
        # enforce particle labels on the final matrices
        if name in ("dm_reco_vs_gen_jet", "dm_reco_vs_gen_tau", "dm_reco_vs_gen_hps"):
            xax = obj.GetXaxis()
            yax = obj.GetYaxis()
            # X: reco, 6 bins (0,1,2,5,10,11)
            for i, lab in enumerate(reco_labels, start=1):
                if i <= xax.GetNbins():
                    xax.SetBinLabel(i, lab)
            # Y: gen, 5 bins (0,1,2,10,11) -> no DM 5
            for j, lab in enumerate(gen_labels, start=1):
                if j <= yax.GetNbins():
                    yax.SetBinLabel(j, lab)

            if name == "dm_reco_vs_gen_jet":
                obj.GetXaxis().SetTitle("Reco DM in jet")
            elif name == "dm_reco_vs_gen_tau":
                obj.GetXaxis().SetTitle("Reco DM in tau")
            else:  # dm_reco_vs_gen_hps
                obj.GetXaxis().SetTitle("DM assigned by HPS")
            obj.GetYaxis().SetTitle("Gen DM")

        draw_hist(obj, out, "COLZ")
    else:
        draw_hist(obj, out, "HIST E1")

# ======= 4) Full tau-level histograms (raw inputs) =======
for dm in dm_list:
    # denominators
    for var in ["pt"] + (["eta"] if PLOT_ETA else []):
        nm = f"tau_dm{dm}_den_{var}"
        obj = d.Get(nm)
        if obj: draw_hist(obj, os.path.join(full_tau_dir, f"{nm}.png"), "HIST E1")
        else:   missing.append(f"{subdir}/{nm}")

    # ≥N charged numerators (N up to prongs)
    for N in range(1, ch_legs_by_dm[dm] + 1):
        for var in ["pt"] + (["eta"] if PLOT_ETA else []):
            nm = f"tau_dm{dm}_ge{N}ch_num_{var}"
            obj = d.Get(nm)
            if obj: draw_hist(obj, os.path.join(full_tau_dir, f"{nm}.png"), "HIST E1")
            else:   missing.append(f"{subdir}/{nm}")

    # ≥N pi0 numerators (N up to pi0 count per DM)
    pi0_cap = {0:0,1:1,2:2,5:0,10:0,11:1}[dm]  # match harvesting_cfg caps
    for N in range(1, pi0_cap + 1):
        for var in ["pt"] + (["eta"] if PLOT_ETA else []):
            nm = f"tau_dm{dm}_ge{N}pi0_num_{var}"
            obj = d.Get(nm)
            if obj: draw_hist(obj, os.path.join(full_tau_dir, f"{nm}.png"), "HIST E1")
            else:   missing.append(f"{subdir}/{nm}")

    # ALL expected numerators (booked for all, filled for DM 1,2,11)
    for var in ["pt"] + (["eta"] if PLOT_ETA else []):
        nm = f"tau_dm{dm}_all_num_{var}"
        obj = d.Get(nm)
        if obj: draw_hist(obj, os.path.join(full_tau_dir, f"{nm}.png"), "HIST E1")
        else:   missing.append(f"{subdir}/{nm}")

# Report missing without failing
if missing:
    print("Missing objects:")
    for m in sorted(set(missing)):
        print("  -", m)

f.Close()
print(f"Done. Trimmed plots saved to: {out_dir}")
