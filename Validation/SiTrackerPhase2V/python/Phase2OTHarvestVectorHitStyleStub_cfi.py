import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

postProcessorHLTVHStubs = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring("HLT/TrackerPhase2OTStubV/Efficiency/Ingredients"),
    efficiency=cms.vstring(
        "effic_pt_barrel 'Stub Efficiency vs p_{T} (Barrel);p_{T} [GeV];Efficiency' eff_numer_pt_barrel eff_denom_pt_barrel",
        "effic_pt_endcap 'Stub Efficiency vs p_{T} (Endcap);p_{T} [GeV];Efficiency' eff_numer_pt_endcap eff_denom_pt_endcap",
        "effic_pt_all 'Stub Efficiency vs p_{T} (All);p_{T} [GeV];Efficiency' eff_numer_pt_all eff_denom_pt_all",
        "effic_eta_barrel 'Stub Efficiency vs #eta (Barrel);#eta;Efficiency' eff_numer_eta_barrel eff_denom_eta_barrel",
        "effic_eta_endcap 'Stub Efficiency vs #eta (Endcap);#eta;Efficiency' eff_numer_eta_endcap eff_denom_eta_endcap",
        "effic_eta_all 'Stub Efficiency vs #eta (All);#eta;Efficiency' eff_numer_eta_all eff_denom_eta_all",
        # Per module category
        "effic_pt_barrel_flat 'Stub Efficiency vs p_{T} (Barrel Flat);p_{T} [GeV];Efficiency' eff_numer_pt_barrel_flat eff_denom_pt_barrel_flat",
        "effic_pt_barrel_tilted 'Stub Efficiency vs p_{T} (Barrel Tilted);p_{T} [GeV];Efficiency' eff_numer_pt_barrel_tilted eff_denom_pt_barrel_tilted",
        "effic_pt_fwd_pos 'Stub Efficiency vs p_{T} (Forward+);p_{T} [GeV];Efficiency' eff_numer_pt_fwd_pos eff_denom_pt_fwd_pos",
        "effic_pt_fwd_neg 'Stub Efficiency vs p_{T} (Forward-);p_{T} [GeV];Efficiency' eff_numer_pt_fwd_neg eff_denom_pt_fwd_neg",
        "effic_eta_barrel_flat 'Stub Efficiency vs #eta (Barrel Flat);#eta;Efficiency' eff_numer_eta_barrel_flat eff_denom_eta_barrel_flat",
        "effic_eta_barrel_tilted 'Stub Efficiency vs #eta (Barrel Tilted);#eta;Efficiency' eff_numer_eta_barrel_tilted eff_denom_eta_barrel_tilted",
        "effic_eta_fwd_pos 'Stub Efficiency vs #eta (Forward+);#eta;Efficiency' eff_numer_eta_fwd_pos eff_denom_eta_fwd_pos",
        "effic_eta_fwd_neg 'Stub Efficiency vs #eta (Forward-);#eta;Efficiency' eff_numer_eta_fwd_neg eff_denom_eta_fwd_neg",
    ),
    resolution=cms.vstring(),
    outputFileName=cms.untracked.string(""),
)

postProcessorHLTVHStubsFakeRate = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring("HLT/TrackerPhase2OTStubV/FakeRate/Ingredients"),
    efficiency=cms.vstring(
        "fakerate_pt_barrel 'Stub Fake Rate vs p_{T} (Barrel);p_{T} [GeV];Fake Rate' fake_numer_pt_barrel fake_denom_pt_barrel fake",
        "fakerate_pt_endcap 'Stub Fake Rate vs p_{T} (Endcap);p_{T} [GeV];Fake Rate' fake_numer_pt_endcap fake_denom_pt_endcap fake",
        "fakerate_pt_all 'Stub Fake Rate vs p_{T} (All);p_{T} [GeV];Fake Rate' fake_numer_pt_all fake_denom_pt_all fake",
        "fakerate_eta_barrel 'Stub Fake Rate vs #eta (Barrel);#eta;Fake Rate' fake_numer_eta_barrel fake_denom_eta_barrel fake",
        "fakerate_eta_endcap 'Stub Fake Rate vs #eta (Endcap);#eta;Fake Rate' fake_numer_eta_endcap fake_denom_eta_endcap fake",
        "fakerate_eta_all 'Stub Fake Rate vs #eta (All);#eta;Fake Rate' fake_numer_eta_all fake_denom_eta_all fake",
        # Per module category
        "fakerate_pt_barrel_flat 'Stub Fake Rate vs p_{T} (Barrel Flat);p_{T} [GeV];Fake Rate' fake_numer_pt_barrel_flat fake_denom_pt_barrel_flat fake",
        "fakerate_pt_barrel_tilted 'Stub Fake Rate vs p_{T} (Barrel Tilted);p_{T} [GeV];Fake Rate' fake_numer_pt_barrel_tilted fake_denom_pt_barrel_tilted fake",
        "fakerate_pt_fwd_pos 'Stub Fake Rate vs p_{T} (Forward+);p_{T} [GeV];Fake Rate' fake_numer_pt_fwd_pos fake_denom_pt_fwd_pos fake",
        "fakerate_pt_fwd_neg 'Stub Fake Rate vs p_{T} (Forward-);p_{T} [GeV];Fake Rate' fake_numer_pt_fwd_neg fake_denom_pt_fwd_neg fake",
        "fakerate_eta_barrel_flat 'Stub Fake Rate vs #eta (Barrel Flat);#eta;Fake Rate' fake_numer_eta_barrel_flat fake_denom_eta_barrel_flat fake",
        "fakerate_eta_barrel_tilted 'Stub Fake Rate vs #eta (Barrel Tilted);#eta;Fake Rate' fake_numer_eta_barrel_tilted fake_denom_eta_barrel_tilted fake",
        "fakerate_eta_fwd_pos 'Stub Fake Rate vs #eta (Forward+);#eta;Fake Rate' fake_numer_eta_fwd_pos fake_denom_eta_fwd_pos fake",
        "fakerate_eta_fwd_neg 'Stub Fake Rate vs #eta (Forward-);#eta;Fake Rate' fake_numer_eta_fwd_neg fake_denom_eta_fwd_neg fake",
    ),
    resolution=cms.vstring(),
    outputFileName=cms.untracked.string(""),
)

# Per-layer efficiency and fake rate ratios.
_perLayerEffStrings = []
_perLayerFakeStrings = []
_perLayerNames = [
    "B1_flat", "B1_tilted", "B2_flat", "B2_tilted",
    "B3_flat", "B3_tilted", "B4_flat", "B4_tilted",
    "B5_flat", "B5_tilted", "B6_flat", "B6_tilted",
    "FwdP_D1", "FwdP_D2", "FwdP_D3", "FwdP_D4", "FwdP_D5",
    "FwdN_D1", "FwdN_D2", "FwdN_D3", "FwdN_D4", "FwdN_D5",
]
_perLayerTitles = [
    "Barrel L1 Flat", "Barrel L1 Tilted", "Barrel L2 Flat", "Barrel L2 Tilted",
    "Barrel L3 Flat", "Barrel L3 Tilted", "Barrel L4 Flat", "Barrel L4 Tilted",
    "Barrel L5 Flat", "Barrel L5 Tilted", "Barrel L6 Flat", "Barrel L6 Tilted",
    "Forward+ Disk 1", "Forward+ Disk 2", "Forward+ Disk 3", "Forward+ Disk 4", "Forward+ Disk 5",
    "Forward- Disk 1", "Forward- Disk 2", "Forward- Disk 3", "Forward- Disk 4", "Forward- Disk 5",
]

for _name, _title in zip(_perLayerNames, _perLayerTitles):
    _perLayerEffStrings.append(
        "effic_pt_{n} 'Stub Efficiency vs p_{{T}} ({t});p_{{T}} [GeV];Efficiency' eff_numer_pt_{n} eff_denom_pt_{n}".format(n=_name, t=_title)
    )
    _perLayerEffStrings.append(
        "effic_eta_{n} 'Stub Efficiency vs #eta ({t});#eta;Efficiency' eff_numer_eta_{n} eff_denom_eta_{n}".format(n=_name, t=_title)
    )
    _perLayerFakeStrings.append(
        "fakerate_pt_{n} 'Stub Fake Rate vs p_{{T}} ({t});p_{{T}} [GeV];Fake Rate' fake_numer_pt_{n} fake_denom_pt_{n} fake".format(n=_name, t=_title)
    )
    _perLayerFakeStrings.append(
        "fakerate_eta_{n} 'Stub Fake Rate vs #eta ({t});#eta;Fake Rate' fake_numer_eta_{n} fake_denom_eta_{n} fake".format(n=_name, t=_title)
    )

postProcessorHLTVHStubsPerLayerEff = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring("HLT/TrackerPhase2OTStubV/Efficiency/PerLayer"),
    efficiency=cms.vstring(_perLayerEffStrings),
    resolution=cms.vstring(),
    outputFileName=cms.untracked.string(""),
)

postProcessorHLTVHStubsPerLayerFake = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring("HLT/TrackerPhase2OTStubV/FakeRate/PerLayer"),
    efficiency=cms.vstring(_perLayerFakeStrings),
    resolution=cms.vstring(),
    outputFileName=cms.untracked.string(""),
)

postProcessorHLTVHStubsSequence = cms.Sequence(
    postProcessorHLTVHStubs
    + postProcessorHLTVHStubsFakeRate
    + postProcessorHLTVHStubsPerLayerEff
    + postProcessorHLTVHStubsPerLayerFake
)
