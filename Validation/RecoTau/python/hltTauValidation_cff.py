import FWCore.ParameterSet.Config as cms

from Validation.RecoTau.TauValidator import TauValidator as _TauValidator

hltTauValidation = _TauValidator(
    recoTauCollection = "hltHpsPFTauProducer",
    genTauCollection = "tauGenJetsSelectorAllHadrons",
    minDeltaR = 0.3,
    outFolder = "HLT/Tau/TauValidation/DeltaR0p3",
    isHLT = True
)

hltTauValidation_deltaR0p25 = hltTauValidation.clone(
    minDeltaR = 0.25,
    outFolder = "HLT/Tau/TauValidation/DeltaR0p25",
)

hltTauValidation_deltaR0p2 = hltTauValidation.clone(
    minDeltaR = 0.2,
    outFolder = "HLT/Tau/TauValidation/DeltaR0p2",
)

hltTauValidation_deltaR0p15 = hltTauValidation.clone(
    minDeltaR = 0.15,
    outFolder = "HLT/Tau/TauValidation/DeltaR0p15",
)

hltTauValidation_deltaR0p1 = hltTauValidation.clone(
    minDeltaR = 0.1,
    outFolder = "HLT/Tau/TauValidation/DeltaR0p1",
)