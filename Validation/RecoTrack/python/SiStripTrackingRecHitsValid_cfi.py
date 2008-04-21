import FWCore.ParameterSet.Config as cms

StripTrackingRecHitsValid = cms.EDFilter("SiStripTrackingRecHitsValid",
    outputFile = cms.untracked.string('striptrackingrechitshisto.root'),
    associatePixel = cms.bool(False),
    ROUList = cms.vstring('TrackerHitsTIBLowTof', 
        'TrackerHitsTIBHighTof', 
        'TrackerHitsTIDLowTof', 
        'TrackerHitsTIDHighTof', 
        'TrackerHitsTOBLowTof', 
        'TrackerHitsTOBHighTof', 
        'TrackerHitsTECLowTof', 
        'TrackerHitsTECHighTof'),
    trajectoryInput = cms.string('TrackRefitter'),
    associateRecoTracks = cms.bool(False),
    #	string trajectoryInput = "rsWithMaterialTracks"
    associateStrip = cms.bool(True)
)


