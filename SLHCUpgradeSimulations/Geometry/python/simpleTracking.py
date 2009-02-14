import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.RecoTracker_cff import *

from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *

ckfTrackCandidates.doSeedingRegionRebuilding = False
#ckfTrackCandidates.TrajectoryBuilder = 'CkfTrajectoryBuilder'
ckfTrackCandidates.TrajectoryBuilder = 'GroupedCkfTrajectoryBuilder'
ckfTrackCandidates.SeedProducer = 'newCombinedSeeds'
ckfTrackCandidates.useHitsSplitting = False
#GroupedCkfTrajectoryBuilder.bestHitOnly = False

from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
ctfWithMaterialTracks.Fitter = 'KFFittingSmoother'
ctfWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

simpleTracking = cms.Sequence(newSeedFromPairs*newSeedFromTriplets*newCombinedSeeds*ckfTrackCandidates*ctfWithMaterialTracks)

from Configuration.StandardSequences.RawToDigi_cff import siPixelDigis,SiStripRawToDigis
from Configuration.StandardSequences.Reconstruction_cff import trackerlocalreco

fullSimpleTracking = cms.Path(siPixelDigis+SiStripRawToDigis
                              +trackerlocalreco
                              +simpleTracking)
