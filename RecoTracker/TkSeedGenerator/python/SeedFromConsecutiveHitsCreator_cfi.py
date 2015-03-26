import FWCore.ParameterSet.Config as cms

SeedFromConsecutiveHitsCreator = cms.PSet(
  ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
  propagator = cms.string('PropagatorWithMaterial'),
#  propagator = cms.string('PropagatorWithMaterialParabolicMf'),
  SeedMomentumForBOFF = cms.double(5.0),
  OriginTransverseErrorMultiplier = cms.double(1.0),
  MinOneOverPtError = cms.double(1.0),
  magneticField = cms.string('ParabolicMf'),
#  magneticField = cms.string(''),
  TTRHBuilder = cms.string('WithTrackAngle'),
  forceKinematicWithRegionDirection = cms.bool(False)
)
