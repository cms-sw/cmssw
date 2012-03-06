import FWCore.ParameterSet.Config as cms

FlexibleKFFittingSmoother = cms.ESProducer("FlexibleKFFittingSmootherESProducer",
    ComponentName = cms.string('FlexibleKFFittingSmoother'),
    standardFitter = cms.string('RKFittingSmoother'),
    looperFitter = cms.string('LooperFittingSmoother'),
)
