import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.RecoTracker_cff import *

from RecoPixelVertexing.Configuration.RecoPixelVertexing_cff import *
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
from EventFilter.RawDataCollector.rawDataCollector_cfi import *
from Configuration.StandardSequences.RawToDigi_cff import *

# new layer list (3/4 pixel seeding) in stepZero
pixellayertriplets.layerList = cms.vstring('BPix1+BPix2+BPix3',
                                                 'BPix1+BPix2+FPix1_pos',
                                                 'BPix1+BPix2+FPix1_neg',
                                                 'BPix1+FPix1_pos+FPix2_pos',
                                                 'BPix1+FPix1_neg+FPix2_neg'
                                                 )
# stepOne seeding from pixel triplets (3/4 pixel seeding)
stepOneTrackCandidateMaker.src = cms.InputTag("stepOneSeedFromTriplets")

from RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff import *
stepOneSeedFromTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
#process.stepOneSeedFromTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.9

stepOneSeedFromTriplets.OrderedHitsFactoryPSet.SeedingLayers = cms.string('stepOneMixedLayerTriplets')

stepOnemixedlayertriplets = RecoTracker.TkSeedingLayers.MixedLayerTriplets_cfi.mixedlayertriplets.clone()
stepOnemixedlayertriplets.ComponentName = cms.string('stepOneMixedLayerTriplets')
stepOnemixedlayertriplets.layerList = cms.vstring('BPix1+BPix2+BPix3',
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

stepOnemixedlayertriplets.BPix.HitProducer = cms.string('newPixelRecHits')
stepOnemixedlayertriplets.FPix.HitProducer = cms.string('newPixelRecHits')
stepOnemixedlayertriplets.TEC.matchedRecHits = cms.InputTag("newStripRecHits","matchedRecHit")
#stepOnemixedlayertriplets.TID.HitProducer = cms.string('newPixelRecHits')
#stepOnemixedlayertriplets.TIB.HitProducer = cms.string('newPixelRecHits')

# new layer list (3/4 pixel seeding) in steptwo
#secTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
#secTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.1
## no need to change using triplets already
#seclayertriplets.layerList = cms.vstring('BPix1+BPix2+BPix3',
#                                                 'BPix1+BPix2+FPix1_pos',
#                                                 'BPix1+BPix2+FPix1_neg',
#                                                 'BPix1+FPix1_pos+FPix2_pos',
#                                                 'BPix1+FPix1_neg+FPix2_neg'
#                                                 )

# stepThree seeding from pixel triplets (3/4 pixel seeding)
## no need to change using triplets already
#thlayertripletsa.layerList = cms.vstring('BPix1+BPix2+BPix3',
#                                         'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg', 
#                                         'BPix3+FPix1_pos+TID1_pos', 'BPix3+FPix1_neg+TID1_neg', 
#                                         'FPix1_pos+FPix2_pos+TEC1_pos', 'FPix1_neg+FPix2_neg+TEC1_neg',
#                                         'FPix2_pos+TID3_pos+TEC1_pos', 'FPix2_neg+TID3_neg+TEC1_neg',
#                                         'FPix2_pos+TEC2_pos+TEC3_pos', 'FPix2_neg+TEC2_neg+TEC3_neg')

#thlayertripletsb.layerList = cms.vstring('BPix2+BPix3+TIB1', 
#        'BPix2+BPix3+TIB2','BPix3+TIB1+TIB2')

#thTrackCandidates.src = cms.InputTag("thSeedFromTriplets")
#thSeedFromTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
#thSeedFromTriplets.OrderedHitsFactoryPSet.SeedingLayers = cms.string('thPixelLayerTriplets')
#
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
# pixel rec hit collection to be used in stepThree
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

########## changing seeding and sequence

# triplet seeding for stepOne (previous seeeding was 'seedFromPair')
firstStep.replace(newSeedFromPairs,stepOneSeedFromTriplets)

# triplet seeding for stepThree (previous seeeding was 'thPLSeeds')
# now step 3 uses triplets
#thirdStep.replace(thPLSeeds,thSeedFromTriplets)

# remove steps 2-5 from iterative tracking
iterTracking_wo2345=iterTracking.copy()
iterTracking_wo2345.remove(secondStep)
iterTracking_wo2345.remove(thirdStep)
iterTracking_wo2345.remove(fourthStep)
iterTracking_wo2345.remove(fifthStep)

# remove iterTrack tracks collection (it comes partially from steps 4-5) and merge4th5thTracks
# replace them with merge2nd3rdTracks tracks collection
trackCollectionMerging_woiterTracksand2345=trackCollectionMerging.copy()
trackCollectionMerging_woiterTracksand2345.remove(iterTracks)
trackCollectionMerging_woiterTracksand2345.remove(merge4th5thTracks)
trackCollectionMerging_woiterTracksand2345.remove(merge2nd3rdTracks)
generalTracks.TrackProducer2 = cms.string('')

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
