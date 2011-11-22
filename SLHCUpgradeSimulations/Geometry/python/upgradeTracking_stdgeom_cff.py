import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.RecoTracker_cff import *

from RecoPixelVertexing.Configuration.RecoPixelVertexing_cff import *
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
from EventFilter.RawDataCollector.rawDataCollector_cfi import *
from Configuration.StandardSequences.RawToDigi_cff import *

from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import *

## use same cuts as in 363 tracking in 440pre6 framework
initialStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.8
initialStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 0.2
initialStepSeeds.RegionFactoryPSet.RegionPSet.nSigmaZ = 3.0
initialStepTrajectoryFilter.filterPset .maxLostHits = 1
# new layer list to rely less on BPIX 1
pixellayertriplets.layerList = cms.vstring( 'BPix1+BPix2+BPix3',
                                            'BPix1+BPix2+FPix1_pos',
                                            'BPix1+BPix2+FPix1_neg',
                                            'BPix1+FPix1_pos+FPix2_pos',
                                            'BPix1+FPix1_neg+FPix2_neg',
                                                 'BPix2+BPix3+FPix1_pos',
                                                 'BPix2+BPix3+FPix1_neg',
                                                 'BPix2+FPix1_pos+FPix2_pos',
                                                 'BPix2+FPix1_neg+FPix2_neg'
                                                 )


## use mixed triplets for step 1 instead of pixel triplets
lowPtTripletStepSeeds.RegionFactoryPSet = cms.PSet(
        RegionPsetFomBeamSpotBlockFixedZ,
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
    )
lowPtTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'lowPtMixedTripletStepSeedLayers'
lowPtTripletStepSeeds.ClusterCheckPSet.PixelClusterCollectionLabel = 'siPixelClusters'
lowPtTripletStepSeeds.ClusterCheckPSet.ClusterCollectionLabel = 'siStripClusters'

#from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
#lowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.ComponentName = 'LowPtClusterShapeSeedComparitor'
lowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.ComponentName = 'none'

lowPtMixedTripletStepSeedLayers = RecoTracker.TkSeedingLayers.MixedLayerTriplets_cfi.mixedlayertriplets.clone()
lowPtMixedTripletStepSeedLayers.ComponentName = cms.string('lowPtMixedTripletStepSeedLayers')
lowPtMixedTripletStepSeedLayers.layerList = cms.vstring('BPix1+BPix2+BPix3',
                                                 'BPix1+BPix2+FPix1_pos',
                                                 'BPix1+BPix2+FPix1_neg',
                                                 'BPix1+BPix3+FPix1_pos',
                                                 'BPix1+BPix3+FPix1_neg',
                                                 'BPix2+BPix3+FPix1_pos',
                                                 'BPix2+BPix3+FPix1_neg',
                                                 'BPix1+BPix2+FPix2_pos',
                                                 'BPix1+BPix2+FPix2_neg',
                                                 'BPix1+FPix1_pos+FPix2_pos',
                                                 'BPix1+FPix1_neg+FPix2_neg',
                                                 'BPix2+FPix1_pos+FPix2_pos',
                                                 'BPix2+FPix1_neg+FPix2_neg',
                                                 'FPix1_pos+FPix2_pos+TEC1_pos',
                                                 'FPix1_neg+FPix2_neg+TEC1_neg',
                                                 'FPix1_pos+FPix2_pos+TEC2_pos',
                                                 'FPix1_neg+FPix2_neg+TEC2_neg',
                                                 'FPix1_pos+TEC1_pos+TEC2_pos',
                                                 'FPix1_neg+TEC1_neg+TEC2_neg',
                                                 'FPix2_pos+TEC1_pos+TEC2_pos',
                                                 'FPix2_neg+TEC1_neg+TEC2_neg',
                                                 'FPix2_pos+TEC1_pos+TEC3_pos',
                                                 'FPix2_neg+TEC1_neg+TEC3_neg',
                                                 'FPix2_pos+TEC2_pos+TEC3_pos',
                                                 'FPix2_neg+TEC2_neg+TEC3_neg'
                                                 )
lowPtMixedTripletStepSeedLayers.TEC.matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit")
lowPtMixedTripletStepSeedLayers.TEC.skipClusters = cms.InputTag('lowPtTripletStepClusters')
#lowPtMixedTripletStepSeedLayers.TEC.useRingSlector = cms.bool(True)
#lowPtMixedTripletStepSeedLayers.TEC.minRing = cms.int32(1)
#lowPtMixedTripletStepSeedLayers.TEC.maxRing = cms.int32(1)

# to avoid 'too many clusters'
initialStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
lowPtTripletStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)

# avoid 'number of triples exceed maximum'
pixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
initialStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
lowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)

### modify regular tracking sequence to use upgrade version
### which is just the first two steps for now
iterTracking.remove(PixelPairStep)
iterTracking.remove(DetachedTripletStep)
iterTracking.remove(MixedTripletStep)
iterTracking.remove(PixelLessStep)
iterTracking.remove(TobTecStep)

newCombinedSeeds.seedCollections = cms.VInputTag(
      cms.InputTag('initialStepSeeds'),
      cms.InputTag('lowPtTripletStepSeeds')
)
