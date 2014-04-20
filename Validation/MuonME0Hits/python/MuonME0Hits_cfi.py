import FWCore.ParameterSet.Config as cms
from Validation.MuonME0Hits.simTrackMatching_cfi import SimTrackMatching

me0HitsValidation = cms.EDAnalyzer("MuonME0Hits",
                                   debug = cms.untracked.bool(False),
                                   folderPath = cms.untracked.string('MuonME0HitsV/ME0HitTask'),
                                   EffSaveRootFile = cms.untracked.bool(False),
                                   EffRootFileName = cms.untracked.string('ME0Hits_ME.root'),
                                   simTrackMatching = SimTrackMatching
)
