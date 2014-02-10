import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.RecoTracker_cff import *

from RecoPixelVertexing.Configuration.RecoPixelVertexing_cff import *
#from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
from EventFilter.RawDataCollector.rawDataCollector_cfi import *
from Configuration.StandardSequences.RawToDigi_cff import *

from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import *

#the quadruplet merger configuration 
# from this PSet the quadruplet merger uses only the layer list so these could probably be removed
from RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff import *
PixelSeedMergerQuadruplets.BPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
PixelSeedMergerQuadruplets.BPix.HitProducer = cms.string("siPixelRecHits" )
PixelSeedMergerQuadruplets.FPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
PixelSeedMergerQuadruplets.FPix.HitProducer = cms.string("siPixelRecHits" )

# new layer list (3/4 pixel seeding) in stepZero
PixelLayerTriplets.layerList = cms.vstring( 'BPix1+BPix2+BPix3',
                                            'BPix2+BPix3+BPix4',
                                            'BPix1+BPix3+BPix4',
                                            'BPix1+BPix2+BPix4',
                                            'BPix2+BPix3+FPix1_pos',
                                            'BPix2+BPix3+FPix1_neg',
                                            'BPix1+BPix2+FPix1_pos',
                                            'BPix1+BPix2+FPix1_neg',
                                            'BPix2+FPix1_pos+FPix2_pos',
                                            'BPix2+FPix1_neg+FPix2_neg',
                                            'BPix1+FPix1_pos+FPix2_pos',
                                            'BPix1+FPix1_neg+FPix2_neg',
                                            'FPix1_pos+FPix2_pos+FPix3_pos',
                                            'FPix1_neg+FPix2_neg+FPix3_neg'
                                                 )
highPtTripletStepSeedLayers.layerList = cms.vstring( 'BPix1+BPix2+BPix3',
                                            'BPix2+BPix3+BPix4',
                                            'BPix1+BPix3+BPix4',
                                            'BPix1+BPix2+BPix4',
                                            'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
                                            'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
                                            'BPix1+BPix3+FPix1_pos', 'BPix1+BPix3+FPix1_neg',
                                            'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
                                            'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
                                            'BPix1+BPix2+FPix2_pos', 'BPix1+BPix2+FPix2_neg',
                                            'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
                                            'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',
                                            'BPix1+FPix1_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix3_neg'
                                                 )
lowPtTripletStepSeedLayers.layerList = cms.vstring( 'BPix1+BPix2+BPix3',
                                            'BPix2+BPix3+BPix4',
                                            'BPix1+BPix3+BPix4',
                                            'BPix1+BPix2+BPix4',
                                            'BPix2+BPix3+FPix1_pos',
                                            'BPix2+BPix3+FPix1_neg',
                                            'BPix1+BPix2+FPix1_pos',
                                            'BPix1+BPix2+FPix1_neg',
                                            'BPix2+FPix1_pos+FPix2_pos',
                                            'BPix2+FPix1_neg+FPix2_neg',
                                            'BPix1+FPix1_pos+FPix2_pos',
                                            'BPix1+FPix1_neg+FPix2_neg',
                                            'FPix1_pos+FPix2_pos+FPix3_pos',
                                            'FPix1_neg+FPix2_neg+FPix3_neg'
                                                 )

## need changes to mixedtriplets step to use for imcreasing high eta efficiency

mixedTripletStepClusters.oldClusterRemovalInfo = cms.InputTag("pixelPairStepClusters")
mixedTripletStepClusters.trajectories = cms.InputTag("pixelPairStepTracks")
mixedTripletStepClusters.overrideTrkQuals = cms.InputTag('pixelPairStepSelector','pixelPairStep')
mixedTripletStepSeedsA.RegionFactoryPSet.RegionPSet.originRadius = 0.02
mixedTripletStepSeedsB.RegionFactoryPSet.RegionPSet.originRadius = 0.02

## new layer list for mixed triplet step
mixedTripletStepSeedLayersA.layerList = cms.vstring('BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
        'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg', 
        'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg', 
        'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
#        'FPix2_pos+FPix3_pos+TEC1_pos', 'FPix2_neg+FPix3_neg+TEC1_neg',
        'FPix3_pos+BPix5+BPix6', 'FPix3_neg+BPix5+BPix6')

#mixedTripletStepSeedLayersB.layerList = cms.vstring('BPix3+BPix4+TIB1', 'BPix3+BPix4+TIB2')
mixedTripletStepSeedLayersB.layerList = cms.vstring('BPix3+BPix4+BPix5', 'BPix3+BPix4+BPix6')

#--->
# disconnect merger for stepOne and step 2 to have triplets merged
highPtTripletStepSeeds.SeedMergerPSet.mergeTriplets = cms.bool(False)
lowPtTripletStepSeeds.SeedMergerPSet.mergeTriplets = cms.bool(False)
pixelPairStepSeeds.SeedMergerPSet.mergeTriplets = cms.bool(False)
mixedTripletStepSeedsA.SeedMergerPSet.mergeTriplets = cms.bool(False)
mixedTripletStepSeedsB.SeedMergerPSet.mergeTriplets = cms.bool(False)
#<---

# to avoid 'too many clusters'
initialStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
highPtTripletStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
lowPtTripletStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
pixelPairStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
mixedTripletStepSeedsA.ClusterCheckPSet.doClusterCheck = cms.bool(False)
mixedTripletStepSeedsB.ClusterCheckPSet.doClusterCheck = cms.bool(False)

# avoid 'number of triples exceed maximum'
pixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
initialStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
highPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
lowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
mixedTripletStepSeedsA.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
mixedTripletStepSeedsB.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
# avoid 'number of pairs exceed maximum'
pixelPairStepSeeds.OrderedHitsFactoryPSet.maxElement =  cms.uint32(0)

# iterative tracking cuts renormalization (alpha's reduced by 7% 
# to take into account the corresponding increase in number_of_layers)
##process.load("RecoTracker.FinalTrackSelectors.TracksWithQuality_cff")
##from RecoTracker.FinalTrackSelectors.TracksWithQuality_cff import *

initialStepSelector.trackSelectors[0].dz_par1 = cms.vdouble(0.605, 4.0) # 0.65
initialStepSelector.trackSelectors[0].dz_par2 = cms.vdouble(0.42, 4.0) # 0.45
initialStepSelector.trackSelectors[0].d0_par1 = cms.vdouble(0.51, 4.0) # 0.55
initialStepSelector.trackSelectors[0].d0_par2 = cms.vdouble(0.51, 4.0) # 0.55
initialStepSelector.trackSelectors[1].dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
initialStepSelector.trackSelectors[1].dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
initialStepSelector.trackSelectors[1].d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
initialStepSelector.trackSelectors[1].d0_par2 = cms.vdouble(0.372, 4.0) # 0.4
initialStepSelector.trackSelectors[2].dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
initialStepSelector.trackSelectors[2].dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
initialStepSelector.trackSelectors[2].d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
initialStepSelector.trackSelectors[2].d0_par2 = cms.vdouble(0.372, 4.0) # 0.4

lowPtTripletStepSelector.trackSelectors[0].dz_par1 = cms.vdouble(0.605, 4.0) # 0.65
lowPtTripletStepSelector.trackSelectors[0].dz_par2 = cms.vdouble(0.42, 4.0) # 0.45
lowPtTripletStepSelector.trackSelectors[0].d0_par1 = cms.vdouble(0.51, 4.0) # 0.55
lowPtTripletStepSelector.trackSelectors[0].d0_par2 = cms.vdouble(0.51, 4.0) # 0.55
lowPtTripletStepSelector.trackSelectors[1].dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
lowPtTripletStepSelector.trackSelectors[1].dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
lowPtTripletStepSelector.trackSelectors[1].d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
lowPtTripletStepSelector.trackSelectors[1].d0_par2 = cms.vdouble(0.372, 4.0) # 0.4
lowPtTripletStepSelector.trackSelectors[2].dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
lowPtTripletStepSelector.trackSelectors[2].dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
lowPtTripletStepSelector.trackSelectors[2].d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
lowPtTripletStepSelector.trackSelectors[2].d0_par2 = cms.vdouble(0.372, 4.0) # 0.4

highPtTripletStepSelector.trackSelectors[0].dz_par1 = cms.vdouble(0.605, 4.0) # 0.65
highPtTripletStepSelector.trackSelectors[0].dz_par2 = cms.vdouble(0.42, 4.0) # 0.45
highPtTripletStepSelector.trackSelectors[0].d0_par1 = cms.vdouble(0.51, 4.0) # 0.55
highPtTripletStepSelector.trackSelectors[0].d0_par2 = cms.vdouble(0.51, 4.0) # 0.55
highPtTripletStepSelector.trackSelectors[1].dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
highPtTripletStepSelector.trackSelectors[1].dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
highPtTripletStepSelector.trackSelectors[1].d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
highPtTripletStepSelector.trackSelectors[1].d0_par2 = cms.vdouble(0.372, 4.0) # 0.4
highPtTripletStepSelector.trackSelectors[2].dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
highPtTripletStepSelector.trackSelectors[2].dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
highPtTripletStepSelector.trackSelectors[2].d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
highPtTripletStepSelector.trackSelectors[2].d0_par2 = cms.vdouble(0.372, 4.0) # 0.4

### modify regular tracking sequence to use upgrade version
### so we can use regular reconstruction step
## remove tracking steps 2-5 to speed up the job
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

