import FWCore.ParameterSet.Config as cms

#Chi2 estimator
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
ElectronChi2 = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = 'ElectronChi2',
    MaxChi2 = 2000.,
    nSigma = 3.,
    MaxDisplacement = 100,
    MaxSagitta = -1
)
# Trajectory Filter
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
TrajectoryFilterForElectrons = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    chargeSignificance = -1.0,
    minPt = 2.0,
    minHitsMinPt = -1,
    ComponentType = 'CkfBaseTrajectoryFilter',
    maxLostHits = 1,
    maxNumberOfHits = -1,
    maxConsecLostHits = 1,
    nSigmaMinPt = 5.0,
    minimumNumberOfHits = 5,
    highEtaSwitch = 2.5,
    minHitsAtHighEta = 3,
    maxCCCLostHits = 9999,
    minGoodStripCharge = dict(refToPSet_ = 'SiStripClusterChargeCutNone')
)

# Phase2 has extended outer-tracker coverage
# so no need to relax cuts on number of hits at high eta 
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(TrajectoryFilterForElectrons, 
    highEtaSwitch = 5.0,
    minHitsAtHighEta = 5
)

# Trajectory Builder
import RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi
TrajectoryBuilderForElectrons = RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi.CkfTrajectoryBuilder.clone(
    trajectoryFilter = dict(refToPSet_ = 'TrajectoryFilterForElectrons'),
    maxCand = 5,
    intermediateCleaning = False,
    propagatorAlong = 'fwdGsfElectronPropagator',
    propagatorOpposite = 'bwdGsfElectronPropagator',
    estimator = 'ElectronChi2',
    lostHitPenalty = 90.,
    alwaysUseInvalidHits = True,
    TTRHBuilder = 'WithTrackAngle',
    updator = 'KFUpdator'
)

# CKFTrackCandidateMaker
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
electronCkfTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'electronMergedSeeds',
    TrajectoryBuilderPSet = dict(refToPSet_ = 'TrajectoryBuilderForElectrons'),
    #TrajectoryCleaner = 'TrajectoryCleanerBySharedHits'
    NavigationSchool = 'SimpleNavigationSchool',
    RedundantSeedCleaner = 'CachingSeedCleanerBySharedInput',
    TrajectoryCleaner = 'electronTrajectoryCleanerBySharedHits'
)

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
electronTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
    ComponentName = 'electronTrajectoryCleanerBySharedHits',
    ValidHitBonus = 1000.0,
    MissingHitPenalty = 0.0
)

# "backward" propagator for electrons
from TrackingTools.GsfTracking.bwdGsfElectronPropagator_cff import *
# "forward" propagator for electrons
from TrackingTools.GsfTracking.fwdGsfElectronPropagator_cff import *
# TrajectoryFilter
