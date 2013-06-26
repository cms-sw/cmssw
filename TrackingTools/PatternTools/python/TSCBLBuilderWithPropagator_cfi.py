import FWCore.ParameterSet.Config as cms

TSCBLBuilderWithPropagator = cms.ESProducer("TSCBLBuilderWithPropagatorESProducer",
ComponentName = cms.string('TSCBLBuilderWithPropagator'),
Propagator = cms.string('AnalyticalPropagator')
)
