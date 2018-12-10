import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
muondtdigianalyzer = DQMEDAnalyzer('MuonDTDigis',
    # Label to retrieve Digis from the event
    DigiLabel = cms.InputTag('simMuonDTDigis'),
    # Label to retrieve SimHits from the event
    SimHitLabel = cms.InputTag('g4SimHits', "MuonDTHits"),
    # Name of the root file which will contain the histos
    outputFile = cms.untracked.string(''),
    # Switch on/off the verbosity
    verbose = cms.untracked.bool(False)
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(muondtdigianalyzer, SimHitLabel = "MuonSimHits:MuonDTHits")
