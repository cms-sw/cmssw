import FWCore.ParameterSet.Config as cms

PixelTrackingRecHitsValid = cms.EDFilter("SiPixelTrackingRecHitsValid",
                                         src = cms.untracked.string('TrackRefitter'),
                                         outputFile = cms.untracked.string('pixeltrackingrechitshist.root'),
                                         Fitter = cms.string('KFFittingSmoother'),
                                         # do we check that the simHit associated with recHit is of the expected particle type ?
                                         checkType = cms.bool(True),
                                         MTCCtrack = cms.bool(False),
                                         TTRHBuilder = cms.string('WithAngleAndTemplate'),
                                         # the type of particle that the simHit associated with recHits should be
                                         genType = cms.int32(13),
                                         associatePixel = cms.bool(True),
                                         associateRecoTracks = cms.bool(False),
                                         associateStrip = cms.bool(False),
                                         ROUList = cms.vstring('g4SimHitsTrackerHitsPixelBarrelLowTof', 
                                                               'g4SimHitsTrackerHitsPixelBarrelHighTof', 
                                                               'g4SimHitsTrackerHitsPixelEndcapLowTof', 
                                                               'g4SimHitsTrackerHitsPixelEndcapHighTof'),
                                         Propagator = cms.string('PropagatorWithMaterial')
                                         )


