import FWCore.ParameterSet.Config as cms

muondtdigianalyzer = cms.EDAnalyzer("MuonDTDigis",
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
if fastSim.isChosen():
    muondtdigianalyzer.SimHitLabel = cms.InputTag("MuonSimHits","MuonDTHits")
