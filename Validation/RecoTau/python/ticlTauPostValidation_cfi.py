import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

dm_list = [0, 1, 2, 5, 10, 11]
effs = []

steps_to_keep = [0, 1, 2, 3, 4, 5]

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

ch_cap = {0: 1, 1: 1, 2: 1, 5: 2, 10: 3, 11: 3}
pi0_cap = {0:0, 1:1, 2:2, 5:0, 10:0, 11:1}
for dm in dm_list:
    cap_ch = ch_cap.get(dm, 0)
    for N in range(1, cap_ch + 1):
        effs.append(
            f"eff_tau_dm{dm}_ge{N}ch_pt  'DM {dm}: >= {N} charged @step5 vs pT'  tau_dm{dm}_ge{N}ch_num_pt  tau_dm{dm}_den_pt"
        )
        effs.append(
            f"eff_tau_dm{dm}_ge{N}ch_eta 'DM {dm}: >= {N} charged @step5 vs eta' tau_dm{dm}_ge{N}ch_num_eta tau_dm{dm}_den_eta"
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