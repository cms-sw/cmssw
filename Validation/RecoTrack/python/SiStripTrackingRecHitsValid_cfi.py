import FWCore.ParameterSet.Config as cms

StripTrackingRecHitsValid = cms.EDAnalyzer("SiStripTrackingRecHitsValid",
    outputFile = cms.untracked.string(''),
    associatePixel = cms.bool(False),
    ROUList = cms.vstring('g4SimHitsTrackerHitsTIBLowTof', 
			'g4SimHitsTrackerHitsTIBHighTof', 
			'g4SimHitsTrackerHitsTIDLowTof', 
			'g4SimHitsTrackerHitsTIDHighTof', 
			'g4SimHitsTrackerHitsTOBLowTof', 
			'g4SimHitsTrackerHitsTOBHighTof', 
			'g4SimHitsTrackerHitsTECLowTof', 
			'g4SimHitsTrackerHitsTECHighTof'),
    trajectoryInput = cms.string('generalTracks'),
    associateRecoTracks = cms.bool(False),
    #	string trajectoryInput = "rsWithMaterialTracks"
    associateStrip = cms.bool(True)
)


