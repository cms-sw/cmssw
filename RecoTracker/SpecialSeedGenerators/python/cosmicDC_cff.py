import FWCore.ParameterSet.Config as cms

# seeding
import RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi 
hitCollectorForCosmicDCSeeds = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName     = 'hitCollectorForCosmicDCSeeds',
    MaxChi2           = 100.0, ## was 30 ## TO BE TUNED
    nSigma            = 4.,    ## was 3  ## TO BE TUNED 
    MaxDisplacement   = 100,
    MaxSagitta        = -1.0,
    MinimalTolerance  = 0.5,
    appendToDataLabel = '',
)
cosmicDCSeeds = RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi.outInSeedsFromStandaloneMuons.clone(
    src          = 'muonsFromCosmics',
    cut          = 'p > 3 && abs(eta)<1.6 && phi<0',
    hitCollector = 'hitCollectorForCosmicDCSeeds',
    fromVertex   = False,
    maxEtaForTOB = 2.5,
    minEtaForTEC = 0.0,
)

# Ckf pattern
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderP5_cff
Chi2MeasurementEstimatorForCDC = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderP5_cff.Chi2MeasurementEstimatorForP5.clone(
    ComponentName   = 'Chi2MeasurementEstimatorForCDC',
    MaxDisplacement = 500,
)

ckfBaseTrajectoryFilterCDC = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderP5_cff.ckfBaseTrajectoryFilterP5.clone(
    maxLostHits       = 10,
    maxConsecLostHits = 10,
)

GroupedCkfTrajectoryBuilderCDC = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderP5_cff.GroupedCkfTrajectoryBuilderP5.clone(
    maxCand   = 3,
    estimator = 'Chi2MeasurementEstimatorForCDC',
    trajectoryFilter = dict(refToPSet_ = 'ckfBaseTrajectoryFilterCDC'),
)

import RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff
cosmicDCCkfTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff.ckfTrackCandidatesP5.clone(
    src = 'cosmicDCSeeds',
    TrajectoryBuilderPSet = dict(refToPSet_ = 'GroupedCkfTrajectoryBuilderCDC'),
)

# Track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff
cosmicDCTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff.ctfWithMaterialTracksCosmics.clone(
    src = 'cosmicDCCkfTrackCandidates',
)

# Final Sequence
cosmicDCTracksSeqTask = cms.Task( cosmicDCSeeds , cosmicDCCkfTrackCandidates , cosmicDCTracks )
cosmicDCTracksSeq = cms.Sequence(cosmicDCTracksSeqTask)
