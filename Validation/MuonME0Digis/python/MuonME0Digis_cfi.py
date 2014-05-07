import FWCore.ParameterSet.Config as cms
from Validation.MuonME0Hits.simTrackMatching_cfi import SimTrackMatching

me0DigiValidation = cms.EDAnalyzer("MuonME0Digis",
                                   debug = cms.untracked.bool(False),
                                   folderPath = cms.untracked.string('MuonME0DigisV/ME0DigiTask'),
                                   EffSaveRootFile = cms.untracked.bool(False),
                                   EffRootFileName = cms.untracked.string('ME0Digis_ME.root'),
                                   simTrackMatching = SimTrackMatching
                                   )
