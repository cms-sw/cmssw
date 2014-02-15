import FWCore.ParameterSet.Config as cms

gemHitsValidation = cms.EDAnalyzer('MuonGEMHits',
    outputFile = cms.string('valid.root'),
    simInputLabel = cms.untracked.string('g4SimHits'),
    verbose = cms.untracked.int32(0),
    simMuOnly = cms.untracked.bool(True),
    discardEleHits = cms.untracked.bool(True),
    minPt = cms.untracked.double(4.5),
    minEta = cms.untracked.double(1.45),
    maxEta = cms.untracked.double(2.5),
)
