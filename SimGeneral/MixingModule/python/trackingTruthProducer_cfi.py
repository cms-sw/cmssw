import FWCore.ParameterSet.Config as cms

trackingParticles = cms.PSet(
	accumulatorType = cms.string('TrackingTruthAccumulator'),
	createUnmergedCollection = cms.bool(True),
	createMergedBremsstrahlung = cms.bool(True),
	createInitialVertexCollection = cms.bool(False),
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
	vertexDistanceCut = cms.double(0.003),
	ignoreTracksOutsideVolume = cms.bool(False),
	allowDifferentSimHitProcesses = cms.bool(False), # should be True for FastSim, False for FullSim
	HepMCProductLabel = cms.InputTag('generatorSmeared')
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    # for unknown reasons, fastsim needs this flag on
    trackingParticles.allowDifferentSimHitProcesses = True
    # fastsim labels for simhits, simtracks, simvertices
    trackingParticles.simHitCollections = cms.PSet(
        muon = cms.VInputTag( cms.InputTag('MuonSimHits','MuonDTHits'),
                              cms.InputTag('MuonSimHits','MuonCSCHits'),
                              cms.InputTag('MuonSimHits','MuonRPCHits') ),
        trackerAndPixel = cms.VInputTag( cms.InputTag('famosSimHits','TrackerHits') )
        )
    trackingParticles.simTrackCollection = cms.InputTag('famosSimHits')
    trackingParticles.simVertexCollection = cms.InputTag('famosSimHits')

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toModify(trackingParticles, simHitCollections = dict(
        muon = trackingParticles.simHitCollections.muon+[cms.InputTag("g4SimHits","MuonGEMHits")]))

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify(trackingParticles, simHitCollections = dict(
        muon = trackingParticles.simHitCollections.muon+[cms.InputTag("g4SimHits","MuonGEMHits")]))

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify( trackingParticles, simHitCollections = dict(
        muon = trackingParticles.simHitCollections.muon+[cms.InputTag("g4SimHits","MuonME0Hits")]))

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify( trackingParticles, simHitCollections = dict( tracker = []) )
