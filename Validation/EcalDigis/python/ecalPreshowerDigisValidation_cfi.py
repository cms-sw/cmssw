import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalPreshowerDigisValidation = DQMEDAnalyzer('EcalPreshowerDigisValidation',
    ESdigiCollection = cms.InputTag("simEcalPreshowerDigis"),
    verbose = cms.untracked.bool(False)
)


