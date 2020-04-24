import FWCore.ParameterSet.Config as cms

ecalEndcapRecHitsValidation = cms.EDAnalyzer("EcalEndcapRecHitsValidation",
    EEdigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
    EEuncalibrechitCollection = cms.InputTag("ecalMultiFitUncalibRecHit","EcalUncalibRecHitsEE"),
    verbose = cms.untracked.bool(False)
)



