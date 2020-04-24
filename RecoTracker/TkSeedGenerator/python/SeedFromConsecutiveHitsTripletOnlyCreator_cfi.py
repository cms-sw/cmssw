import FWCore.ParameterSet.Config as cms
SeedFromConsecutiveHitsTripletOnlyCreator = cms.PSet(
  ComponentName = cms.string('SeedFromConsecutiveHitsTripletOnlyCreator'),
  propagator = cms.string('PropagatorWithMaterial')
#  propagator = cms.string('PropagatorWithMaterialParabolicMf')
)
