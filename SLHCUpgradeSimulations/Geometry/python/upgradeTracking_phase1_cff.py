import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.RecoTracker_cff import *

from RecoPixelVertexing.Configuration.RecoPixelVertexing_cff import *
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
from EventFilter.RawDataCollector.rawDataCollector_cfi import *
from Configuration.StandardSequences.RawToDigi_cff import *

#the quadruplet merger configuration 
from RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff import *
pixelseedmergerlayers.BPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
pixelseedmergerlayers.BPix.HitProducer = cms.string("siPixelRecHits" )
pixelseedmergerlayers.FPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
pixelseedmergerlayers.FPix.HitProducer = cms.string("siPixelRecHits" )

# new layer list (3/4 pixel seeding) in stepZero
pixellayertriplets.layerList = cms.vstring( 'BPix1+BPix2+BPix3',
                                            'BPix1+BPix2+BPix4',
                                            'BPix2+BPix3+BPix4',
                                            'BPix1+BPix3+BPix4',
                                            'BPix1+BPix2+FPix1_pos', 
                                            'BPix1+BPix2+FPix1_neg', 
                                            'BPix1+FPix1_pos+FPix2_pos', 
                                            'BPix1+FPix1_neg+FPix2_neg',
                                            'BPix1+FPix2_pos+FPix3_pos',
                                            'BPix1+FPix2_neg+FPix3_neg'
                                                 )
# stepOne seeding from pixel triplets (3/4 pixel seeding)
stepOneTrackCandidateMaker.src = cms.InputTag("stepOneSeedFromTriplets")

from RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff import *
stepOneSeedFromTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
#stepOneSeedFromTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.9

stepOneSeedFromTriplets.OrderedHitsFactoryPSet.SeedingLayers = cms.string('stepOneMixedLayerTriplets')

#--->
# disconnect merger for stepOne to have triplets
stepOneSeedFromTriplets.SeedMergerPSet.mergeTriplets = cms.bool(False)
#<---

stepOnemixedlayertriplets = RecoTracker.TkSeedingLayers.MixedLayerTriplets_cfi.mixedlayertriplets.clone()
stepOnemixedlayertriplets.ComponentName = cms.string('stepOneMixedLayerTriplets')
stepOnemixedlayertriplets.layerList = cms.vstring('BPix1+BPix2+BPix3',
                                                 'BPix1+BPix2+BPix4',
                                                 'BPix2+BPix3+BPix4',
                                                 'BPix1+BPix3+BPix4',
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
                                                 'FPix1_neg+FPix2_neg+FPix3_neg',
                                                 'FPix2_pos+FPix3_pos+TEC1_pos',
                                                 'FPix2_neg+FPix3_neg+TEC1_neg',
                                                 'FPix2_pos+FPix3_pos+TEC2_pos',
                                                 'FPix2_neg+FPix3_neg+TEC2_neg',
                                                 'FPix2_pos+TEC1_pos+TEC2_pos',
                                                 'FPix2_neg+TEC1_neg+TEC2_neg',
                                                 'FPix3_pos+TEC1_pos+TEC2_pos',
                                                 'FPix3_neg+TEC1_neg+TEC2_neg',
                                                 'FPix3_pos+TEC1_pos+TEC3_pos',
                                                 'FPix3_neg+TEC1_neg+TEC3_neg',
                                                 'FPix3_pos+TEC2_pos+TEC3_pos',
                                                 'FPix3_neg+TEC2_neg+TEC3_neg'
                                                 )

stepOnemixedlayertriplets.BPix.HitProducer = cms.string('newPixelRecHits')
stepOnemixedlayertriplets.FPix.HitProducer = cms.string('newPixelRecHits')
stepOnemixedlayertriplets.TEC.matchedRecHits = cms.InputTag("newStripRecHits","matchedRecHit")
#stepOnemixedlayertriplets.TID.HitProducer = cms.string('newPixelRecHits')
#stepOnemixedlayertriplets.TIB.HitProducer = cms.string('newPixelRecHits')


# new layer list (3/4 pixel seeding) in steptwo
#secTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
#secTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.1
seclayertriplets.layerList = cms.vstring( 'BPix1+BPix2+BPix3',
                                          'BPix2+BPix3+BPix4',
                                          'BPix1+BPix2+FPix1_pos',
                                          'BPix1+BPix2+FPix1_neg',
                                          'BPix1+FPix1_pos+FPix2_pos',
                                          'BPix1+FPix1_neg+FPix2_neg',
                                          'FPix1_pos+FPix2_pos+FPix3_pos',
                                          'FPix1_neg+FPix2_neg+FPix3_neg'
                                                 )
                                                 
# stepThree seeding from pixel triplets (3/4 pixel seeding)
thlayertripletsa.layerList = cms.vstring('BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
        'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
        'BPix3+FPix1_pos+TID1_pos', 'BPix3+FPix1_neg+TID1_neg',
        'BPix4+FPix1_pos+TID1_pos', 'BPix4+FPix1_neg+TID1_neg',
        'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg'
        'FPix1_pos+FPix2_pos+TEC1_pos', 'FPix1_neg+FPix2_neg+TEC1_neg',
        'FPix2_pos+FPix3_pos+TEC1_pos', 'FPix2_neg+FPix3_neg+TEC1_neg',
        'FPix2_pos+FPix3_pos+TID3_pos', 'FPix2_neg+FPix3_neg+TID3_neg',
        'FPix2_pos+FPix3_pos+TEC2_pos', 'FPix2_neg+FPix3_neg+TEC2_neg',
        'FPix2_pos+TID3_pos+TEC1_pos', 'FPix2_neg+TID3_neg+TEC1_neg',
        'FPix2_pos+TEC2_pos+TEC3_pos', 'FPix2_neg+TEC2_neg+TEC3_neg',
        'FPix3_pos+TID3_pos+TEC1_pos', 'FPix3_neg+TID3_neg+TEC1_neg',
        'FPix3_pos+TEC2_pos+TEC3_pos', 'FPix3_neg+TEC2_neg+TEC3_neg')
                                                 
thlayertripletsb.layerList = cms.vstring('BPix2+BPix3+BPix4','BPix3+BPix4+TIB1',
        'BPix3+BPix4+TIB2','BPix4+TIB1+TIB2')

#--->
# disconnect merger for stepThree to have triplets
thTripletsA.SeedMergerPSet.mergeTriplets = cms.bool(False)
thTripletsB.SeedMergerPSet.mergeTriplets = cms.bool(False)
#<---

# stepThree seeding from pixel triplets (3/4 pixel seeding)
#thTrackCandidates.src = cms.InputTag("thSeedFromTriplets")
#thSeedFromTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
#thSeedFromTriplets.OrderedHitsFactoryPSet.SeedingLayers = cms.string('thPixelLayerTriplets')

#--->
# disconnect merger for stepThree to have triplets
#thSeedFromTriplets.SeedMergerPSet.mergeTriplets = cms.bool(False)
#<---

#thpixellayertriplets = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.pixellayertriplets.clone()
#thpixellayertriplets.ComponentName = cms.string('thPixelLayerTriplets')
#thpixellayertriplets.layerList = cms.vstring('BPix1+BPix2+BPix3',
#                                                 'BPix2+BPix3+BPix4',
#                                                 'BPix1+BPix2+FPix1_pos',
#                                                 'BPix1+BPix2+FPix1_neg',
#                                                 'BPix1+BPix3+FPix1_pos',
#                                                 'BPix1+BPix3+FPix1_neg',
#                                                 'BPix2+BPix3+FPix1_pos',
#                                                 'BPix2+BPix3+FPix1_neg',
#                                                 'BPix1+BPix2+FPix2_pos',
#                                                 'BPix1+BPix2+FPix2_neg',
#                                                 'BPix1+FPix1_pos+FPix2_pos',
#                                                 'BPix1+FPix1_neg+FPix2_neg',
#                                                 'BPix1+FPix2_pos+FPix3_pos',
#                                                 'BPix1+FPix2_neg+FPix3_neg',
#                                                 'FPix1_pos+FPix2_pos+FPix3_pos',
#                                                 'FPix1_neg+FPix2_neg+FPix3_neg'
#                                                 )
#                                             
## pixel rec hit collection to be used in stepThree
#thpixellayertriplets.BPix.HitProducer = cms.string('thPixelRecHits')
#thpixellayertriplets.FPix.HitProducer = cms.string('thPixelRecHits')

# to avoid 'too many clusters'
newSeedFromTriplets.ClusterCheckPSet.doClusterCheck = cms.bool(False)
stepOneSeedFromTriplets.ClusterCheckPSet.doClusterCheck = cms.bool(False)
secTriplets.ClusterCheckPSet.doClusterCheck = cms.bool(False)
#thSeedFromTriplets.ClusterCheckPSet.doClusterCheck = cms.bool(False)
thTripletsA.ClusterCheckPSet.doClusterCheck = cms.bool(False)
thTripletsB.ClusterCheckPSet.doClusterCheck = cms.bool(False)
fourthPLSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
fifthSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)

# avoid 'number of triples exceed maximum'
pixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
newSeedFromTriplets.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
stepOneSeedFromTriplets.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
secTriplets.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
thTripletsA.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
thTripletsB.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
#thSeedFromTriplets.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)

# iterative tracking cuts renormalization (alpha's reduced by 7% 
# to take into account the corresponding increase in number_of_layers)
##process.load("RecoTracker.FinalTrackSelectors.TracksWithQuality_cff")
##from RecoTracker.FinalTrackSelectors.TracksWithQuality_cff import *

zeroStepWithLooseQuality.dz_par1 = cms.vdouble(0.605, 4.0) # 0.65
zeroStepWithLooseQuality.dz_par2 = cms.vdouble(0.42, 4.0) # 0.45
zeroStepWithLooseQuality.d0_par1 = cms.vdouble(0.51, 4.0) # 0.55
zeroStepWithLooseQuality.d0_par2 = cms.vdouble(0.51, 4.0) # 0.55
zeroStepWithTightQuality.dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
zeroStepWithTightQuality.dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
zeroStepWithTightQuality.d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
zeroStepWithTightQuality.d0_par2 = cms.vdouble(0.372, 4.0) # 0.4
zeroStepTracksWithQuality.dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
zeroStepTracksWithQuality.dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
zeroStepTracksWithQuality.d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
zeroStepTracksWithQuality.d0_par2 = cms.vdouble(0.372, 4.0) # 0.4

firstStepWithLooseQuality.dz_par1 = cms.vdouble(0.605, 4.0) # 0.65
firstStepWithLooseQuality.dz_par2 = cms.vdouble(0.42, 4.0) # 0.45
firstStepWithLooseQuality.d0_par1 = cms.vdouble(0.51, 4.0) # 0.55
firstStepWithLooseQuality.d0_par2 = cms.vdouble(0.51, 4.0) # 0.55
firstStepWithTightQuality.dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
firstStepWithTightQuality.dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
firstStepWithTightQuality.d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
firstStepWithTightQuality.d0_par2 = cms.vdouble(0.372, 4.0) # 0.4
preMergingFirstStepTracksWithQuality.dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
preMergingFirstStepTracksWithQuality.dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
preMergingFirstStepTracksWithQuality.d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
preMergingFirstStepTracksWithQuality.d0_par2 = cms.vdouble(0.372, 4.0) # 0.4

secStepVtxLoose.dz_par1 = cms.vdouble(1.116, 3.0) # 1.2
secStepVtxLoose.dz_par2 = cms.vdouble(1.209, 3.0) # 1.3
secStepVtxLoose.d0_par1 = cms.vdouble(1.116, 3.0) # 1.2
secStepVtxLoose.d0_par2 = cms.vdouble(1.209, 3.0) # 1.3
secStepVtxTight.dz_par1 = cms.vdouble(0.837, 3.0) # 0.9
secStepVtxTight.dz_par2 = cms.vdouble(0.93, 3.0) # 1.0
secStepVtxTight.d0_par1 = cms.vdouble(0.8835, 3.0) # 0.95
secStepVtxTight.d0_par2 = cms.vdouble(0.93, 3.0) # 1.0
secStepVtx.dz_par1 = cms.vdouble(0.744, 3.0) # 0.8
secStepVtx.dz_par2 = cms.vdouble(0.837, 3.0) # 0.9
secStepVtx.d0_par1 = cms.vdouble(0.7905, 3.0) # 0.85
secStepVtx.d0_par2 = cms.vdouble(0.837, 3.0) # 0.9

secStepTrkLoose.dz_par1 = cms.vdouble(1.395, 4.0) # 1.5
secStepTrkLoose.dz_par2 = cms.vdouble(1.395, 4.0) # 1.5
secStepTrkLoose.d0_par1 = cms.vdouble(1.395, 4.0) # 1.5
secStepTrkLoose.d0_par2 = cms.vdouble(1.395, 4.0) # 1.5
secStepTrkTight.dz_par1 = cms.vdouble(0.93, 4.0) # 1.1
secStepTrkTight.dz_par2 = cms.vdouble(0.93, 4.0) # 1.
secStepTrkTight.d0_par1 = cms.vdouble(0.93, 4.0) # 1.
secStepTrkTight.d0_par2 = cms.vdouble(0.93, 4.0) # 1.
secStepTrk.dz_par1 = cms.vdouble(0.837, 4.0) # 0.9
secStepTrk.dz_par2 = cms.vdouble(0.837, 4.0) # 0.9
secStepTrk.d0_par1 = cms.vdouble(0.837, 4.0) # 0.9
secStepTrk.d0_par2 = cms.vdouble(0.837, 4.0) # 0.9

#thStepVtxLoose.dz_par1 = cms.vdouble(1.116, 3.0) # 1.2
#thStepVtxLoose.dz_par2 = cms.vdouble(1.209, 3.0) # 1.3
#thStepVtxLoose.d0_par1 = cms.vdouble(1.116, 3.0) # 1.2
#thStepVtxLoose.d0_par2 = cms.vdouble(1.209, 3.0) # 1.3
#thStepVtxTight.dz_par1 = cms.vdouble(0.93, 3.0) # 1.0
#thStepVtxTight.dz_par2 = cms.vdouble(1.023, 3.0) # 1.1
#thStepVtxTight.d0_par1 = cms.vdouble(0.93, 3.0) # 1.0
#thStepVtxTight.d0_par2 = cms.vdouble(1.023, 3.0) # 1.1
#thStepVtx.dz_par1 = cms.vdouble(0.837, 3.0) # 0.9
#thStepVtx.dz_par2 = cms.vdouble(0.93, 3.0) # 1.0
#thStepVtx.d0_par1 = cms.vdouble(0.837, 3.0) # 0.9
#thStepVtx.d0_par2 = cms.vdouble(0.93, 3.0) # 1.0

#thStepTrkLoose.dz_par1 = cms.vdouble(1.674, 4.0) # 1.8
#thStepTrkLoose.dz_par2 = cms.vdouble(1.674, 4.0) # 1.8
#thStepTrkLoose.d0_par1 = cms.vdouble(1.674, 4.0) # 1.8
#thStepTrkLoose.d0_par2 = cms.vdouble(1.674, 4.0) # 1.8
#thStepTrkTight.dz_par1 = cms.vdouble(1.023, 4.0) # 1.1
#thStepTrkTight.dz_par2 = cms.vdouble(1.023, 4.0) # 1.1
#thStepTrkTight.d0_par1 = cms.vdouble(1.023, 4.0) # 1.1
#thStepTrkTight.d0_par2 = cms.vdouble(1.023, 4.0) # 1.1
#thStepTrk.dz_par1 = cms.vdouble(0.93, 4.0) # 1.0
#thStepTrk.dz_par2 = cms.vdouble(0.93, 4.0) # 1.0
#thStepTrk.d0_par1 = cms.vdouble(0.93, 4.0) # 1.0
#thStepTrk.d0_par2 = cms.vdouble(0.93, 4.0) # 1.0

MeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
newMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
secMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
thMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
fourthMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
fifthMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()

########## changing seeding and sequence

# triplet seeding for stepOne (previous seeeding was 'seedFromPair')
firstStep.replace(newSeedFromPairs,stepOneSeedFromTriplets) 

# triplet seeding for stepThree (previous seeeding was 'thPLSeeds')
#thirdStep.replace(thTriplets,thSeedFromTriplets) 

# remove steps 4 and 5 from iterative tracking
#iterTracking_wo45=iterTracking.copy()
#iterTracking_wo45.remove(fourthStep)
#iterTracking_wo45.remove(fifthStep)
# remove steps 2-5 from iterative tracking
iterTracking_wo2345=iterTracking.copy()
iterTracking_wo2345.remove(secondStep)
iterTracking_wo2345.remove(thirdStep)
iterTracking_wo2345.remove(fourthStep)
iterTracking_wo2345.remove(fifthStep)

# remove iterTrack tracks collection (it comes partially from steps 4-5) and merge4th5thTracks
# replace them with merge2nd3rdTracks tracks collection
#trackCollectionMerging_woiterTracksand45=trackCollectionMerging.copy()
#trackCollectionMerging_woiterTracksand45.remove(iterTracks)
#trackCollectionMerging_woiterTracksand45.remove(merge4th5thTracks)
#generalTracks.TrackProducer2 = 'merge2nd3rdTracks'
# remove iterTrack tracks collection (it comes partially from steps 4-5) and merge4th5thTracks
# replace them with merge2nd3rdTracks tracks collection
trackCollectionMerging_woiterTracksand2345=trackCollectionMerging.copy()
trackCollectionMerging_woiterTracksand2345.remove(iterTracks)
trackCollectionMerging_woiterTracksand2345.remove(merge4th5thTracks)
trackCollectionMerging_woiterTracksand2345.remove(merge2nd3rdTracks)
generalTracks.TrackProducer2 = cms.string('')

# change combined seed (pair seed no more implemented) leave it for backward compatibility with electron reconstruction
#ckftracks_wodEdXandSteps4and5 = ckftracks_wodEdX.copy()
#newCombinedSeeds.seedCollections = cms.VInputTag(
#  cms.InputTag('newSeedFromTriplets'),
#  cms.InputTag('stepOneSeedFromTriplets'),
#)
#ckftracks_wodEdXandCombinedSeeds.remove(newCombinedSeeds)
#ckftracks_wodEdXandSteps4and5.replace(iterTracking,iterTracking_wo45)
#ckftracks_wodEdXandSteps4and5.replace(trackCollectionMerging,trackCollectionMerging_woiterTracksand45)
#
# change combined seed (pair seed no more implemented) leave it for backward compatibility with electron reconstruction
ckftracks_wodEdXandSteps2345 = ckftracks_wodEdX.copy()
newCombinedSeeds.seedCollections = cms.VInputTag(
  cms.InputTag('newSeedFromTriplets'),
  cms.InputTag('stepOneSeedFromTriplets'),
)
ckftracks_wodEdXandSteps2345.replace(iterTracking,iterTracking_wo2345)
ckftracks_wodEdXandSteps2345.replace(trackCollectionMerging,trackCollectionMerging_woiterTracksand2345)

### modify regular tracking sequence to use upgrade version
### so we can use regular reconstruction step
iterTracking.remove(secondStep)
iterTracking.remove(thirdStep)
iterTracking.remove(fourthStep)
iterTracking.remove(fifthStep)
trackCollectionMerging.remove(iterTracks)
trackCollectionMerging.remove(merge4th5thTracks)
trackCollectionMerging.remove(merge2nd3rdTracks)
generalTracks.TrackProducer2 = cms.string('')
newCombinedSeeds.seedCollections = cms.VInputTag(
  cms.InputTag('newSeedFromTriplets'),
  cms.InputTag('stepOneSeedFromTriplets'),
)

