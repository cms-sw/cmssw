import FWCore.ParameterSet.Config as cms
from Validation.MuonME0RecHits.simTrackMatching_cfi import SimTrackMatching

me0RecHitsValidation = cms.EDAnalyzer("MuonME0RecHits",
                                   debug = cms.untracked.bool(False),
                                   folderPath = cms.untracked.string('MuonME0RecHitsV/ME0RecHitTask'),
                                   EffSaveRootFile = cms.untracked.bool(False),
                                   EffRootFileName = cms.untracked.string('ME0RecHits_ME.root'),
                                   simTrackMatching = SimTrackMatching
)
