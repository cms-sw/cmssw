import FWCore.ParameterSet.Config as cms
SeedFromConsecutiveHitsStraightLineCreator = cms.PSet(
  ComponentName = cms.string('SeedFromConsecutiveHitsStraightLineCreator'),
  propagator = cms.string('PropagatorWithMaterial')
)
