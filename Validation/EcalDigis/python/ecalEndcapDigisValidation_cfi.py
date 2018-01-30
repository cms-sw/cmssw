import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalEndcapDigisValidation = DQMEDAnalyzer('EcalEndcapDigisValidation',
    EEdigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
    verbose = cms.untracked.bool(False)
)


