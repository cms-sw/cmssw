import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.RecoTracker_cff import *

from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *

ckfTrackCandidates.doSeedingRegionRebuilding = False
#ckfTrackCandidates.TrajectoryBuilder = 'CkfTrajectoryBuilder'
ckfTrackCandidates.TrajectoryBuilder = 'GroupedCkfTrajectoryBuilder'
#ckfTrackCandidates.SeedProducer = 'newCombinedSeeds'
ckfTrackCandidates.useHitsSplitting = False
#GroupedCkfTrajectoryBuilder.bestHitOnly = False

from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
ctfWithMaterialTracks.Fitter = 'KFFittingSmoother'
#ctfWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

from RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cfi import *
from RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cfi import *
###ckfTrackCandidates.SeedProducer = 'globalMixedSeeds'

###oldTracking = cms.Sequence(globalMixedSeeds*globalPixelSeeds*ckfTrackCandidates*ctfWithMaterialTracks)


from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *
pixelTriplets = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    #include "RecoTracker/PixelStubs/data/SeedComparitorWithPixelStubs.cfi"
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.string('PixelLayerTriplets'),
        GeneratorPSet = cms.PSet(
            PixelTripletHLTGenerator
        )
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
ckfTrackCandidates.SeedProducer = 'pixelTriplets'
oldTracking_wtriplets = cms.Sequence(pixelTriplets*ckfTrackCandidates*ctfWithMaterialTracks)



from Configuration.StandardSequences.RawToDigi_cff import siPixelDigis,SiStripRawToDigis
from Configuration.StandardSequences.Reconstruction_cff import trackerlocalreco


fullOldTracking_wtriplets = cms.Path(siPixelDigis+SiStripRawToDigis
                              +trackerlocalreco
                              +pixelTriplets
                              +oldTracking_wtriplets)
