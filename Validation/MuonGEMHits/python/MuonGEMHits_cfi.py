import FWCore.ParameterSet.Config as cms



from Validation.MuonGEMHits.simTrackMatching_cfi import SimTrackMatching
from Validation.MuonGEMHits.gemSystemSetting_cfi import gemSetting
gemHitsValidation = cms.EDAnalyzer('MuonGEMHits',
	outputFile = cms.string(''),
        simInputLabel = cms.untracked.string('g4SimHits'),
	minPt = cms.untracked.double(4.5),
	ntupleTrackChamberDelta = cms.untracked.bool(True),
	ntupleTrackEff = cms.untracked.bool(True),        
        simTrackMatching = SimTrackMatching,
        gemSystemSetting = gemSetting
)
