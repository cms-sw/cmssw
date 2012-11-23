import FWCore.ParameterSet.Config as cms
SeedFromConsecutiveHitsStraightLineCreator = cms.PSet(
  ComponentName = cms.string('SeedFromConsecutiveHitsStraightLineCreator'),
  propagator = cms.string('PropagatorWithMaterial'),
  OriginTransverseErrorMultiplier = cms.double(1.0),
  MinOneOverPtError = cms.double(1.0)
)
