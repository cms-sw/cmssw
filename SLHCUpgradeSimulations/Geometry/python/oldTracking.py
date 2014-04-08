import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.RecoTracker_cff import *

from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *

ckfTrackCandidates.doSeedingRegionRebuilding = False
#ckfTrackCandidates.TrajectoryBuilderPSet.refToPSet_ = 'CkfTrajectoryBuilder'
ckfTrackCandidates.TrajectoryBuilderPSet.refToPSet_ = 'GroupedCkfTrajectoryBuilder'
#ckfTrackCandidates.SeedProducer = 'newCombinedSeeds'
ckfTrackCandidates.useHitsSplitting = False
#GroupedCkfTrajectoryBuilder.bestHitOnly = False

from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
ctfWithMaterialTracks.Fitter = 'KFFittingSmoother'
#ctfWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

from RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cfi import *
from RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cfi import *
ckfTrackCandidates.SeedProducer = 'globalMixedSeeds'

oldTracking = cms.Sequence(globalMixedSeeds*globalPixelSeeds*ckfTrackCandidates*ctfWithMaterialTracks)

from Configuration.StandardSequences.RawToDigi_cff import siPixelDigis,SiStripRawToDigis
from Configuration.StandardSequences.Reconstruction_cff import trackerlocalreco

fullOldTracking = cms.Path(siPixelDigis+SiStripRawToDigis
                              +trackerlocalreco
                              +oldTracking)
