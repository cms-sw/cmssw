import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.RecoTracker_cff import *

from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *

ckfTrackCandidates.doSeedingRegionRebuilding = False
#ckfTrackCandidates.TrajectoryBuilder.refToPSet_ = 'CkfTrajectoryBuilder'
ckfTrackCandidates.TrajectoryBuilderPSet.refToPSet_ = 'GroupedCkfTrajectoryBuilder'
#ckfTrackCandidates.SeedProducer = 'newCombinedSeeds'
ckfTrackCandidates.useHitsSplitting = False
#GroupedCkfTrajectoryBuilder.bestHitOnly = False

from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
ctfWithMaterialTracks.Fitter = 'KFFittingSmoother'
#ctfWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

from RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff import *
###ckfTrackCandidates.SeedProducer = 'globalMixedSeeds'

###oldTracking = cms.Sequence(globalMixedSeeds*globalPixelSeeds*ckfTrackCandidates*ctfWithMaterialTracks)


from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *
from RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff import PixelSeedMergerQuadruplets
pixelTriplets = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    #include "RecoTracker/PixelStubs/data/SeedComparitorWithPixelStubs.cfi"
    ClusterCheckPSet = cms.PSet( 
        MaxNumberOfCosmicClusters = cms.uint32( 50000 ),
        ClusterCollectionLabel = cms.InputTag( "siStripClusters" ),
        doClusterCheck = cms.bool( False ) #so this isn't used
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.InputTag('PixelLayerTriplets'),
        GeneratorPSet = cms.PSet(
            PixelTripletHLTGenerator
        )
    ),
     SeedMergerPSet = cms.PSet(
        # layer list for the merger, as defined in (or modified from):
        # RecoPixelVertexing/PixelTriplets/python/quadrupletseedmerging_cff.py
        layerList = PixelSeedMergerQuadruplets,
        # merge triplets -> quadruplets if applicable?
        mergeTriplets = cms.bool( True ),
        # add remaining (non-merged) triplets to merged output quadruplets?
        # (results in a "mixed" output)
        addRemainingTriplets = cms.bool( False ),
        # the builder
        ttrhBuilderLabel = cms.string( "PixelTTRHBuilderWithoutAngle" )
    ),
     SeedCreatorPSet = cms.PSet( 
        ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
        propagator = cms.string( "PropagatorWithMaterial" )
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        RegionPSetBlock,
        ComponentName = cms.string('GlobalRegionProducer')
    ),
    SeedMomentumForBOFF = cms.double(5.0),
    TTRHBuilder = cms.string('WithTrackAngle')
)
ckfTrackCandidates.src = 'pixelTriplets'
#ckfTrackCandidates.SeedProducer = 'pixelTriplets'
oldTracking_wtriplets = cms.Sequence(PixelLayerTriplets*pixelTriplets*ckfTrackCandidates*ctfWithMaterialTracks)



from Configuration.StandardSequences.RawToDigi_cff import siPixelDigis #,SiStripRawToDigis
from Configuration.StandardSequences.Reconstruction_cff import trackerlocalreco

#print "siPixelDigis",siPixelDigis.label
#print "SiStripRawToDigis",SiStripRawToDigis.label
#print "trackerlocalreco",trackerlocalreco.label
#print "pixelTriplets",pixelTriplets.label
#print "oldTracking_wtriplets",oldTracking_wtriplets.label
#print "ckfTrackCandidates",ckfTrackCandidates.label
#print "ctfWithMaterialTracks",ctfWithMaterialTracks.label

fullOldTracking_wtriplets = cms.Path(siPixelDigis #+SiStripRawToDigis
                              +trackerlocalreco
                            #  +pixelTriplets
                              +oldTracking_wtriplets)
