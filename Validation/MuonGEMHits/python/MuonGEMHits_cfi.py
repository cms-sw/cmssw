import FWCore.ParameterSet.Config as cms


gemHitsValidation = cms.EDAnalyzer('MuonGEMHits',
    outputFile = cms.string(''),
    verbose = cms.untracked.int32(0),
    simInputLabel = cms.untracked.string('g4SimHits'),
    ntupleTrackChamberDelta = cms.untracked.bool(True),
    ntupleTrackEff = cms.untracked.bool(True),        
    simMuOnly = cms.untracked.bool(True),
    discardEleHits = cms.untracked.bool(True),
    simTrackMatching = cms.PSet( 
       gemMinPt = cms.untracked.double(4.5),
       gemMinEta = cms.untracked.double(1.45),
       gemMaxEta = cms.untracked.double(2.5)
    )
)

