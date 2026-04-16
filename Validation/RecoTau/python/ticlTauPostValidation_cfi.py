import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

dm_list = [0, 1, 2, 10, 11]
effs = []

steps_to_keep = [0, 1, 2, 3, 4, 5]
track_steps = [0, 1, 2, 3, 4]
combi_steps = [0, 1, 2]

for dm in dm_list:
    for leg in range(3):
        for s in steps_to_keep:
            effs.append(
                f"eff_ch_dm{dm}_leg{leg}_step{s}_pt  "
                f"'DM {dm} ch leg{leg} step{s}: efficiency vs pT'  "
                f"ch_dm{dm}_leg{leg}_step{s}_num_pt  ch_dm{dm}_leg{leg}_step{s}_den_pt"
            )
            effs.append(
                f"eff_ch_dm{dm}_leg{leg}_step{s}_eta "
                f"'DM {dm} ch leg{leg} step{s}: efficiency vs eta' "
                f"ch_dm{dm}_leg{leg}_step{s}_num_eta ch_dm{dm}_leg{leg}_step{s}_den_eta"
            )
        # track chain (charged hadrons only)
        for s in track_steps:
            effs.append(
                f"eff_ch_dm{dm}_leg{leg}_trkstep{s}_pt  "
                f"'DM {dm} ch leg{leg} trkStep{s}: efficiency vs pT'  "
                f"ch_dm{dm}_leg{leg}_trkstep{s}_num_pt  ch_dm{dm}_leg{leg}_trkstep{s}_den_pt"
            )
            effs.append(
                f"eff_ch_dm{dm}_leg{leg}_trkstep{s}_eta "
                f"'DM {dm} ch leg{leg} trkStep{s}: efficiency vs eta' "
                f"ch_dm{dm}_leg{leg}_trkstep{s}_num_eta ch_dm{dm}_leg{leg}_trkstep{s}_den_eta"
            )
        # combi (AND) chain (charged hadrons only)
        for s in combi_steps:
            effs.append(
                f"eff_ch_dm{dm}_leg{leg}_combistep{s}_pt  "
                f"'DM {dm} ch leg{leg} combiStep{s}: efficiency vs pT'  "
                f"ch_dm{dm}_leg{leg}_combistep{s}_num_pt  ch_dm{dm}_leg{leg}_combistep{s}_den_pt"
            )
            effs.append(
                f"eff_ch_dm{dm}_leg{leg}_combistep{s}_eta "
                f"'DM {dm} ch leg{leg} combiStep{s}: efficiency vs eta' "
                f"ch_dm{dm}_leg{leg}_combistep{s}_num_eta ch_dm{dm}_leg{leg}_combistep{s}_den_eta"
            )
    for leg in range(4):
        for s in steps_to_keep:
            effs.append(
                f"eff_pho_dm{dm}_leg{leg}_step{s}_pt  "
                f"'DM {dm} pho leg{leg} step{s}: efficiency vs pT'  "
                f"pho_dm{dm}_leg{leg}_step{s}_num_pt  pho_dm{dm}_leg{leg}_step{s}_den_pt"
            )
            effs.append(
                f"eff_pho_dm{dm}_leg{leg}_step{s}_eta "
                f"'DM {dm} pho leg{leg} step{s}: efficiency vs eta' "
                f"pho_dm{dm}_leg{leg}_step{s}_num_eta pho_dm{dm}_leg{leg}_step{s}_den_eta"
            )

ch_cap = {0: 1, 1: 1, 2: 1, 10: 3, 11: 3}
pi0_cap = {0:0, 1:1, 2:2, 10:0, 11:1}
for dm in dm_list:
    cap_ch = ch_cap.get(dm, 0)
    for N in range(1, cap_ch + 1):
        effs.append(
            f"eff_tau_dm{dm}_ge{N}ch_pt  'DM {dm}: >= {N} charged @step5 vs pT'  tau_dm{dm}_ge{N}ch_num_pt  tau_dm{dm}_den_pt"
        )
        effs.append(
            f"eff_tau_dm{dm}_ge{N}ch_eta 'DM {dm}: >= {N} charged @step5 vs eta' tau_dm{dm}_ge{N}ch_num_eta tau_dm{dm}_den_eta"
        )
        # tau-level two-fold: track / calo / both / either
        for kind, tag in [("track", "track-matched"), ("calo", "calo-matched"),
                          ("both", "track AND calo"), ("either", "track OR calo")]:
            effs.append(
                f"eff_tau_dm{dm}_ge{N}ch_{kind}_pt  'DM {dm}: >= {N} charged {tag} vs pT'  "
                f"tau_dm{dm}_ge{N}ch_{kind}_num_pt  tau_dm{dm}_den_pt"
            )
            effs.append(
                f"eff_tau_dm{dm}_ge{N}ch_{kind}_eta 'DM {dm}: >= {N} charged {tag} vs eta' "
                f"tau_dm{dm}_ge{N}ch_{kind}_num_eta tau_dm{dm}_den_eta"
            )
    cap_p0 = pi0_cap.get(dm, 0)
    for N in range(1, cap_p0 + 1):
        effs.append(
            f"eff_tau_dm{dm}_ge{N}pi0_pt  'DM {dm}: >= {N} pi0 @step5 vs pT'  tau_dm{dm}_ge{N}pi0_num_pt  tau_dm{dm}_den_pt"
        )
        effs.append(
            f"eff_tau_dm{dm}_ge{N}pi0_eta 'DM {dm}: >= {N} pi0 @step5 vs eta' tau_dm{dm}_ge{N}pi0_num_eta tau_dm{dm}_den_eta"
        )
    if dm in (1, 2, 11):
        effs.append(
            f"eff_tau_dm{dm}_all_pt  'DM {dm}: ALL expected @step5 vs pT'  tau_dm{dm}_all_num_pt  tau_dm{dm}_den_pt"
        )
        effs.append(
            f"eff_tau_dm{dm}_all_eta 'DM {dm}: ALL expected @step5 vs eta' tau_dm{dm}_all_num_eta tau_dm{dm}_den_eta"
        )
    # Two-fold CP-level efficiencies (charged: trackOnly, caloOnly, trackAndCalo)
    for kind in ["trackOnly", "caloOnly", "trackAndCalo"]:
        effs.append(
            f"eff_cp_chHad_dm{dm}_{kind}_pt  "
            f"'DM {dm} charged CP {kind} vs pT'  "
            f"cp_chHad_dm{dm}_{kind}_pt  cp_chHad_dm{dm}_pt"
        )
        effs.append(
            f"eff_cp_chHad_dm{dm}_{kind}_eta "
            f"'DM {dm} charged CP {kind} vs eta' "
            f"cp_chHad_dm{dm}_{kind}_eta cp_chHad_dm{dm}_eta"
        )
    # Two-fold CP-level efficiencies (photon: caloOnly)
    effs.append(
        f"eff_cp_gamma_dm{dm}_caloOnly_pt  "
        f"'DM {dm} photon CP caloOnly vs pT'  "
        f"cp_gamma_dm{dm}_caloOnly_pt  cp_gamma_dm{dm}_pt"
    )
    effs.append(
        f"eff_cp_gamma_dm{dm}_caloOnly_eta "
        f"'DM {dm} photon CP caloOnly vs eta' "
        f"cp_gamma_dm{dm}_caloOnly_eta cp_gamma_dm{dm}_eta"
    )

# Fake rate efficiencies (num / den = fake rate)
fake_effs = []

# Inclusive fake rate
fake_effs.append(
    "fake_rate_pt  'Fake rate vs pT'  fake_num_pt  fake_den_pt"
)
fake_effs.append(
    "fake_rate_eta 'Fake rate vs eta' fake_num_eta fake_den_eta"
)

# Per reco DM fake rate (DMs 0,1,2,5,10,11)
fake_dm_list = [0, 1, 2, 5, 10, 11]
for dm in fake_dm_list:
    fake_effs.append(
        f"fake_rate_dm{dm}_pt  'DM {dm}: fake rate vs pT'  fake_dm{dm}_num_pt  fake_dm{dm}_den_pt"
    )
    fake_effs.append(
        f"fake_rate_dm{dm}_eta 'DM {dm}: fake rate vs eta' fake_dm{dm}_num_eta fake_dm{dm}_den_eta"
    )

# Calo-only and track-only fake rates (same structure, different prefix)
for assoc, tag in [("calo", "calo assoc"), ("track", "track assoc")]:
    prefix = f"fake_{assoc}"
    fake_effs.append(
        f"{prefix}_rate_pt  'Fake rate ({tag}) vs pT'  {prefix}_num_pt  {prefix}_den_pt"
    )
    fake_effs.append(
        f"{prefix}_rate_eta 'Fake rate ({tag}) vs eta' {prefix}_num_eta {prefix}_den_eta"
    )
    for dm in fake_dm_list:
        fake_effs.append(
            f"{prefix}_rate_dm{dm}_pt  'DM {dm}: fake rate ({tag}) vs pT'  {prefix}_dm{dm}_num_pt  {prefix}_dm{dm}_den_pt"
        )
        fake_effs.append(
            f"{prefix}_rate_dm{dm}_eta 'DM {dm}: fake rate ({tag}) vs eta' {prefix}_dm{dm}_num_eta {prefix}_dm{dm}_den_eta"
        )

# Charged iso path: combined, calo-only, track-only
for assoc_prefix, assoc_tag in [("fake_chargedIsoPath", "charged iso path"),
                                 ("fake_calo_chargedIsoPath", "calo assoc, charged iso path"),
                                 ("fake_track_chargedIsoPath", "track assoc, charged iso path")]:
    fake_effs.append(
        f"{assoc_prefix}_rate_pt  'Fake rate ({assoc_tag}) vs pT'  {assoc_prefix}_num_pt  {assoc_prefix}_den_pt"
    )
    fake_effs.append(
        f"{assoc_prefix}_rate_eta 'Fake rate ({assoc_tag}) vs eta' {assoc_prefix}_num_eta {assoc_prefix}_den_eta"
    )
    for dm in fake_dm_list:
        fake_effs.append(
            f"{assoc_prefix}_rate_dm{dm}_pt  'DM {dm}: fake rate ({assoc_tag}) vs pT'  {assoc_prefix}_dm{dm}_num_pt  {assoc_prefix}_dm{dm}_den_pt"
        )
        fake_effs.append(
            f"{assoc_prefix}_rate_dm{dm}_eta 'DM {dm}: fake rate ({assoc_tag}) vs eta' {assoc_prefix}_dm{dm}_num_eta {assoc_prefix}_dm{dm}_den_eta"
        )

effs += fake_effs

# Client (folder must match analyzer fill path: "SimTauValidator")
recoTiclTauHarvester = DQMEDHarvester(
    "DQMGenericClient",
    verbose=cms.untracked.uint32(1),
    runOnEndLumi=cms.untracked.bool(False),
    runOnEndJob=cms.untracked.bool(True),
    makeGlobalEffienciesPlot=cms.untracked.bool(False),

    # matches recoTiclTauValidator.folder
    subDirs=cms.untracked.vstring("RecoTauV/ticlTauValidator/*"),

    efficiency=cms.vstring(*effs),
    resolution=cms.vstring(),
    efficiencyProfile=cms.untracked.vstring(),
    resolutionProfile=cms.untracked.vstring(),
    profile=cms.untracked.vstring(),
    normalization=cms.untracked.vstring(),
    cumulativeDists=cms.untracked.vstring(),
    noFlowDists=cms.untracked.vstring(),
    resolutionLimitedFit=cms.untracked.bool(False),
    outputFileName=cms.untracked.string(""),
)

hltTiclTauHarvester = DQMEDHarvester(
    "DQMGenericClient",
    verbose=cms.untracked.uint32(5),
    runOnEndLumi=cms.untracked.bool(False),
    runOnEndJob=cms.untracked.bool(True),
    makeGlobalEffienciesPlot=cms.untracked.bool(False),
    subDirs=cms.untracked.vstring("HLT/TICL/ticlTauValidator/*"),
    efficiency=cms.vstring(*effs),
    resolution=cms.vstring(),
    efficiencyProfile=cms.untracked.vstring(),
    resolutionProfile=cms.untracked.vstring(),
    profile=cms.untracked.vstring(),
    normalization=cms.untracked.vstring(),
    cumulativeDists=cms.untracked.vstring(),
    noFlowDists=cms.untracked.vstring(),
    resolutionLimitedFit=cms.untracked.bool(False),
    outputFileName=cms.untracked.string(""),
)

ticlTauHarvesting = cms.Sequence(
    hltTiclTauHarvester
)


#ticlTauHarvesting = cms.Sequence(
#    hltTiclTauHarvester +
#    recoTiclTauHarvester
#)