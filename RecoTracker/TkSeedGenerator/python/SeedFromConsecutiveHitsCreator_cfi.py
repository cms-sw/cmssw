import FWCore.ParameterSet.Config as cms

SeedFromConsecutiveHitsCreator = cms.PSet(
  ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
#  propagator = cms.string('PropagatorWithMaterial'),
  propagator = cms.string('PropagatorWithMaterialParabolicMf'),
  SeedMomentumForBOFF = cms.double(5.0),
  OriginTransverseErrorMultiplier = cms.double(1.0),
  MinOneOverPtError = cms.double(1.0),
  magneticField = cms.string('ParabolicMf'),
#  magneticField = cms.string(''),
  TTRHBuilder = cms.string('WithTrackAngle'),
  forceKinematicWithRegionDirection = cms.bool(False)
)
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1PU70.toModify(SeedFromConsecutiveHitsCreator,
        magneticField = '',
        propagator = 'PropagatorWithMaterial'
)
