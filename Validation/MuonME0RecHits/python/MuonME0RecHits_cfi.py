import FWCore.ParameterSet.Config as cms
from Validation.MuonME0RecHits.simTrackMatching_cfi import SimTrackMatching

me0RecHitsValidation = cms.EDAnalyzer("MuonME0RecHits",
                                   debug = cms.untracked.bool(True),
#                                   ME0SimHitInput = cms.untracked.InputTag('g4SimHits','MuonME0Hits'),
#                                   SimTrackInput = cms.untracked.InputTag('g4SimHits'),
                                   folderPath = cms.untracked.string('MuonME0RecHitsV/ME0RecHitTask'),
                                   EffSaveRootFile = cms.untracked.bool(True),
                                   EffRootFileName = cms.untracked.string('ME0RecHits_ME.root'),
                                   simTrackMatching = SimTrackMatching
)
