import FWCore.ParameterSet.Config as cms

RecHitAnalyzer = cms.EDAnalyzer("RecHitAnalyzer",
                      debug = cms.untracked.bool(True),
                      folderPath = cms.untracked.string('GEMBasicPlots/'),
                      EffSaveRootFile = cms.untracked.bool(True),
                      EffRootFileName = cms.untracked.string('GEMRecHit_ME.root')
)
