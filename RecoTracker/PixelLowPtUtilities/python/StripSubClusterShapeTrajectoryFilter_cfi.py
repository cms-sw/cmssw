import FWCore.ParameterSet.Config as cms

from RecoTracker.PixelLowPtUtilities.StripSubClusterShapeFilter_cfi import StripSubClusterShapeFilterParams
StripSubClusterShapeTrajectoryFilter = cms.PSet(
    StripSubClusterShapeFilterParams,
    ComponentType = cms.string('StripSubClusterShapeTrajectoryFilter'),
)

StripSubClusterShapeTrajectoryFilterTIB12 = cms.PSet(
    StripSubClusterShapeTrajectoryFilter,
    layerMask = cms.PSet(
        TIB = cms.vuint32(1,2),
        TOB = cms.bool(False),
        TID = cms.bool(False),
        TEC = cms.bool(False),
    ),
)

StripSubClusterShapeTrajectoryFilterTIX12 = cms.PSet(
    StripSubClusterShapeTrajectoryFilter,
    layerMask = cms.PSet(
        TIB = cms.vuint32(1,2),
        TOB = cms.bool(False),
        TID = cms.vuint32(1,2),
        TEC = cms.bool(False),
    ),
)
