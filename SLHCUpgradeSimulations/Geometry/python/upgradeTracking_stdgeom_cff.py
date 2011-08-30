import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.RecoTracker_cff import *

from RecoPixelVertexing.Configuration.RecoPixelVertexing_cff import *
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
from EventFilter.RawDataCollector.rawDataCollector_cfi import *
from Configuration.StandardSequences.RawToDigi_cff import *

## use mixed triplets for step 1 instead of pixel triplets
lowPtTripletStepSeedLayers.layerList = cms.vstring('BPix1+BPix2+BPix3',
                                                 'BPix1+BPix2+FPix1_pos',
                                                 'BPix1+BPix2+FPix1_neg',
                                                 'BPix1+BPix3+FPix1_pos',
                                                 'BPix1+BPix3+FPix1_neg',
                                                 'BPix2+BPix3+FPix1_pos',
                                                 'BPix2+BPix3+FPix1_neg',
                                                 'BPix1+BPix2+FPix2_pos',
                                                 'BPix1+BPix2+FPix2_neg',
                                                 'BPix1+BPix2+FPix3_pos',
                                                 'BPix1+BPix2+FPix3_neg',
                                                 'BPix1+FPix1_pos+FPix2_pos',
                                                 'BPix1+FPix1_neg+FPix2_neg',
                                                 'BPix1+FPix1_pos+FPix3_pos',
                                                 'BPix1+FPix1_neg+FPix3_neg',
                                                 'BPix1+FPix2_pos+FPix3_pos',
                                                 'BPix1+FPix2_neg+FPix3_neg',
                                                 'BPix2+FPix1_pos+FPix2_pos',
                                                 'BPix2+FPix1_neg+FPix2_neg',
                                                 'BPix2+FPix1_pos+FPix3_pos',
                                                 'BPix2+FPix1_neg+FPix3_neg',
                                                 'BPix2+FPix2_pos+FPix3_pos',
                                                 'BPix2+FPix2_neg+FPix3_neg',
                                                 'FPix1_pos+FPix2_pos+FPix3_pos',
                                                 'FPix1_neg+FPix2_neg+FPix3_neg'
                                                 )
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
