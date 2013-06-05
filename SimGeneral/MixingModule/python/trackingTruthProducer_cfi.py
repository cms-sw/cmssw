import FWCore.ParameterSet.Config as cms

trackingParticles = cms.PSet(
	accumulatorType = cms.string('TrackingTruthAccumulator'),
	createUnmergedCollection = cms.bool(True),
	createMergedBremsstrahlung = cms.bool(True),
	alwaysAddAncestors = cms.bool(True),
	maximumPreviousBunchCrossing = cms.uint32(9999),
	maximumSubsequentBunchCrossing = cms.uint32(9999),
	simHitCollections = cms.PSet(
		muon = cms.VInputTag( cms.InputTag('g4SimHits','MuonDTHits'),
			cms.InputTag('g4SimHits','MuonCSCHits'),
			cms.InputTag('g4SimHits','MuonRPCHits') ),
		tracker = cms.VInputTag( cms.InputTag('g4SimHits','TrackerHitsTIBLowTof'),
			cms.InputTag('g4SimHits','TrackerHitsTIBHighTof'),
			cms.InputTag('g4SimHits','TrackerHitsTIDLowTof'),
			cms.InputTag('g4SimHits','TrackerHitsTIDHighTof'),
			cms.InputTag('g4SimHits','TrackerHitsTOBLowTof'),
			cms.InputTag('g4SimHits','TrackerHitsTOBHighTof'),
			cms.InputTag('g4SimHits','TrackerHitsTECLowTof'),
			cms.InputTag('g4SimHits','TrackerHitsTECHighTof') ),
		pixel = cms.VInputTag(cms.InputTag( 'g4SimHits','TrackerHitsPixelBarrelLowTof'),
        	cms.InputTag('g4SimHits','TrackerHitsPixelBarrelHighTof'),
        	cms.InputTag('g4SimHits','TrackerHitsPixelEndcapLowTof'),
        	cms.InputTag('g4SimHits','TrackerHitsPixelEndcapHighTof') )
	),
	simTrackCollection = cms.InputTag('g4SimHits'),
	simVertexCollection = cms.InputTag('g4SimHits'),
	genParticleCollection = cms.InputTag('genParticles'),
	removeDeadModules = cms.bool(False), # currently not implemented
	volumeRadius = cms.double(120.0),
	volumeZ = cms.double(300.0),
	ignoreTracksOutsideVolume = cms.bool(False),
	allowDifferentSimHitProcesses = cms.bool(False) # should be True for FastSim, False for FullSim
)
