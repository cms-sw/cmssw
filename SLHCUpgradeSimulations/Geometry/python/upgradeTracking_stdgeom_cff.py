import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.RecoTracker_cff import *

from RecoPixelVertexing.Configuration.RecoPixelVertexing_cff import *
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
from EventFilter.RawDataCollector.rawDataCollector_cfi import *
from Configuration.StandardSequences.RawToDigi_cff import *

from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import *

## need changes to mixedtriplets step to use for imcreasing high eta efficiency

mixedTripletStepClusters.oldClusterRemovalInfo = cms.InputTag("pixelPairStepClusters")
mixedTripletStepClusters.trajectories = cms.InputTag("pixelPairStepTracks")
mixedTripletStepClusters.overrideTrkQuals = cms.InputTag('pixelPairStepSelector','pixelPairStep')
mixedTripletStepSeedsA.RegionFactoryPSet.RegionPSet.originRadius = 0.02
mixedTripletStepSeedsB.RegionFactoryPSet.RegionPSet.originRadius = 0.02
## switch off SeedB the easy way
mixedTripletStepSeedLayersB.layerList = cms.vstring('BPix1+BPix2+BPix3')
## increased the max track candidates 
#process.load("RecoTracker.CkfPattern.CkfTrackCandidates_cff")
#process.ckfTrackCandidates.maxNSeeds = cms.uint32(500000)
mixedTripletStepTrackCandidates.maxNSeeds = cms.uint32(150000)
pixelPairStepTrackCandidates.maxNSeeds = cms.uint32(150000)

generalTracks.TrackProducers = (cms.InputTag('initialStepTracks'),
                      cms.InputTag('lowPtTripletStepTracks'),
                      cms.InputTag('pixelPairStepTracks'),
                      cms.InputTag('mixedTripletStepTracks'))
generalTracks.hasSelector=cms.vint32(1,1,1,1)
generalTracks.selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStepSelector","initialStep"),
                                       cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"),
                                       cms.InputTag("pixelPairStepSelector","pixelPairStep"),
                                       cms.InputTag("mixedTripletStep"),
                                       )
generalTracks.setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3), pQual=cms.bool(True) )
                             )


# to avoid 'too many clusters'
initialStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
lowPtTripletStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
pixelPairStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
mixedTripletStepSeedsA.ClusterCheckPSet.doClusterCheck = cms.bool(False)
mixedTripletStepSeedsB.ClusterCheckPSet.doClusterCheck = cms.bool(False)

# avoid 'number of triples exceed maximum'
pixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
initialStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
lowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
mixedTripletStepSeedsA.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
mixedTripletStepSeedsB.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
# avoid 'number of pairs exceed maximum'
pixelPairStepSeeds.OrderedHitsFactoryPSet.maxElement =  cms.uint32(0)

### modify regular tracking sequence to use upgrade version
### which is just the first two steps for now
#iterTracking.remove(PixelPairStep)
iterTracking.remove(DetachedTripletStep)
#iterTracking.remove(MixedTripletStep)
iterTracking.remove(PixelLessStep)
iterTracking.remove(TobTecStep)

#newCombinedSeeds.seedCollections = cms.VInputTag(
#      cms.InputTag('initialStepSeeds'),
#      cms.InputTag('lowPtTripletStepSeeds'),
#      cms.InputTag('pixelPairStepSeeds')
#)
