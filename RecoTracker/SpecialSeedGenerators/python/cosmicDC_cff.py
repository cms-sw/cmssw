import FWCore.ParameterSet.Config as cms

# seeding
import RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi 
hitCollectorForCosmicDCSeeds = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('hitCollectorForCosmicDCSeeds'),
    MaxChi2 = cms.double(100.0), ## was 30 ## TO BE TUNED
    nSigma  = cms.double(4.),    ## was 3  ## TO BE TUNED 
    MaxDisplacement = cms.double(100),
    MaxSagitta = cms.double(-1.0),
    MinimalTolerance = cms.double(0.5),
    appendToDataLabel = cms.string(''),
)
cosmicDCSeeds = RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi.outInSeedsFromStandaloneMuons.clone(
    src = cms.InputTag("muonsFromCosmics"),
    cut = cms.string("p > 3 && abs(eta)<1.6 && phi<0"),
    hitCollector = cms.string('hitCollectorForCosmicDCSeeds'),
    fromVertex = cms.bool(False),
    maxEtaForTOB = cms.double(2.5),
    minEtaForTEC = cms.double(0.0),
)

# Ckf pattern
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderP5_cff
Chi2MeasurementEstimatorForCDC = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderP5_cff.Chi2MeasurementEstimatorForP5.clone(
    ComponentName = cms.string('Chi2MeasurementEstimatorForCDC'),
    MaxDisplacement = 500
)

ckfBaseTrajectoryFilterCDC = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderP5_cff.ckfBaseTrajectoryFilterP5.clone(
    maxLostHits = 10,
    maxConsecLostHits = 10
)

GroupedCkfTrajectoryBuilderCDC = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderP5_cff.GroupedCkfTrajectoryBuilderP5.clone(
    maxCand = 3,
    estimator = 'Chi2MeasurementEstimatorForCDC',
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('ckfBaseTrajectoryFilterCDC')
    )
)

import RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff
cosmicDCCkfTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff.ckfTrackCandidatesP5.clone(
    src = cms.InputTag( "cosmicDCSeeds" ),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('GroupedCkfTrajectoryBuilderCDC')
    )
)

# Track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff
cosmicDCTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff.ctfWithMaterialTracksCosmics.clone(
    src = cms.InputTag( "cosmicDCCkfTrackCandidates" ),
)

# Final Sequence
cosmicDCTracksSeqTask = cms.Task( cosmicDCSeeds , cosmicDCCkfTrackCandidates , cosmicDCTracks )
cosmicDCTracksSeq = cms.Sequence(cosmicDCTracksSeqTask)
