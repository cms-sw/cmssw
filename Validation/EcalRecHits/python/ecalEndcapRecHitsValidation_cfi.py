import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalEndcapRecHitsValidation = DQMEDAnalyzer('EcalEndcapRecHitsValidation',
    EEdigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
    EEuncalibrechitCollection = cms.InputTag("ecalMultiFitUncalibRecHit","EcalUncalibRecHitsEE"),
    verbose = cms.untracked.bool(False)
)



# foo bar baz
# 1hxpGRT4EpxeM
# ypYJDejE6B8EL
