import FWCore.ParameterSet.Config as cms

gemHitsValidation = cms.EDAnalyzer('MuonGEMHits',
    outputFile = cms.string(''),
    verbose = cms.untracked.int32(0),
    simInputLabel = cms.untracked.string('g4SimHits'),
    ntupleTrackChamberDelta = cms.untracked.bool(True),
    ntupleTrackEff = cms.untracked.bool(True),        
    simMuOnly = cms.untracked.bool(True),
    discardEleHits = cms.untracked.bool(True),
    minPt = cms.untracked.double(4.5),
    minEta = cms.untracked.double(1.45),
    maxEta = cms.untracked.double(2.5),
)
