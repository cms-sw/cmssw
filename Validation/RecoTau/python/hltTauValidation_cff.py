import FWCore.ParameterSet.Config as cms

from Validation.RecoTau.TauValidator import TauValidator as _TauValidator

hltTauValidation = _TauValidator(
    genTauCollection = "tauGenJetsSelectorAllHadrons", # only GenTaus decaying hadronically
    recoTauCollection = "hltHpsPFTauProducer",
    recoTauIDCollections = ["hltHpsPFTauDeepTauProducer:VSjet", "hltHpsPFTauDeepTauProducer:VSe", "hltHpsPFTauDeepTauProducer:VSmu"],
    cutIDs_wp = [-1, -1, -1], # WP discriminator (disabled if < 0)
    cutIDs_raw = [0.0, 0.0, 0.0], # raw discriminator value cuts (disabled if 0.0)
    minDeltaR = 0.3,
    outFolder = "HLT/Tau/TauValidation",
    isPatTaus = False
)

hltTauValidation_cutWPVsJet_0 = hltTauValidation.clone(cutIDs_wp = [0, -1, -1])
hltTauValidation_cutWPVsJet_1 = hltTauValidation.clone(cutIDs_wp = [1, -1, -1])
hltTauValidation_cutWPVsJet_2 = hltTauValidation.clone(cutIDs_wp = [2, -1, -1])
hltTauValidation_cutWPVsJet_3 = hltTauValidation.clone(cutIDs_wp = [3, -1, -1])

hltTauValidation_cutIdVsJet_0p5 = hltTauValidation.clone(cutIDs_raw = [0.5, -1, -1])
hltTauValidation_cutIdVsJet_0p7 = hltTauValidation.clone(cutIDs_raw = [0.7, -1, -1])
hltTauValidation_cutIdVsJet_0p9 = hltTauValidation.clone(cutIDs_raw = [0.9, -1, -1])
hltTauValidation_cutIdVsJet_0p95 = hltTauValidation.clone(cutIDs_raw = [0.95, -1, -1])
hltTauValidation_cutIdVsJet_0p99 = hltTauValidation.clone(cutIDs_raw = [0.99, -1, -1])

hltTauValidation_deltaR0p3 = hltTauValidation.clone(
    minDeltaR = 0.3,
    outFolder = "HLT/Tau/TauValidation_DeltaR/DeltaR0p3",
)

hltTauValidation_deltaR0p25 = hltTauValidation.clone(
    minDeltaR = 0.25,
    outFolder = "HLT/Tau/TauValidation_DeltaR/DeltaR0p25",
)

hltTauValidation_deltaR0p2 = hltTauValidation.clone(
    minDeltaR = 0.2,
    outFolder = "HLT/Tau/TauValidation_DeltaR/DeltaR0p2",
)

hltTauValidation_deltaR0p15 = hltTauValidation.clone(
    minDeltaR = 0.15,
    outFolder = "HLT/Tau/TauValidation_DeltaR/DeltaR0p15",
)

hltTauValidation_deltaR0p1 = hltTauValidation.clone(
    minDeltaR = 0.1,
    outFolder = "HLT/Tau/TauValidation_DeltaR/DeltaR0p1",
)

hltTauValidation_deltaR = cms.Sequence(
    hltTauValidation_deltaR0p3
    + hltTauValidation_deltaR0p25
    + hltTauValidation_deltaR0p2
    + hltTauValidation_deltaR0p15
    + hltTauValidation_deltaR0p1
)

hltTauValidation_wp = cms.Sequence(
    hltTauValidation_cutWPVsJet_0
    + hltTauValidation_cutWPVsJet_1
    + hltTauValidation_cutWPVsJet_2
    + hltTauValidation_cutWPVsJet_3
)

hltTauValidation_id = cms.Sequence(
    hltTauValidation_cutIdVsJet_0p5
    + hltTauValidation_cutIdVsJet_0p7
    + hltTauValidation_cutIdVsJet_0p9
    + hltTauValidation_cutIdVsJet_0p95
    + hltTauValidation_cutIdVsJet_0p99
)

hltTauValidationSequence = cms.Sequence(
    hltTauValidation
    # WP scanning
    + hltTauValidation_wp
    # ID scanning
    + hltTauValidation_id
    # DeltaR scanning
    + hltTauValidation_deltaR
)