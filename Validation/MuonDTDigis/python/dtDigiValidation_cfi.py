import FWCore.ParameterSet.Config as cms

muondtdigianalyzer = cms.EDAnalyzer("MuonDTDigis",
    # Label to retrieve Digis from the event
    DigiLabel = cms.untracked.string('simMuonDTDigis'),
    # Label to retrieve SimHits from the event
    SimHitLabel = cms.untracked.string('g4SimHits'),
    # Name of the root file which will contain the histos
    outputFile = cms.untracked.string(''),
    # Switch on/off the verbosity
    verbose = cms.untracked.bool(False) 
)



