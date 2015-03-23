Bimport FWCore.ParameterSet.Config as cms
0;95;c
SeedFromConsecutiveHitsCreator = cms.PSet(
  ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
#  propagator = cms.string('PropagatorWithMaterial'),
  propagator = cms.string('PropagatorWithMaterialParabolicMf'),
  SeedMomentumForBOFF = cms.double(5.0),
  OriginTransverseErrorMultiplier = cms.double(1.0),
  MinOneOverPtError = cms.double(1.0),
  SimpleMagneticField = cms.string('ParabolicMf'),
  TTRHBuilder = cms.string('WithTrackAngle')
)
