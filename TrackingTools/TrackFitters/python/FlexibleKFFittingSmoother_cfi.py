import FWCore.ParameterSet.Config as cms

FlexibleKFFittingSmoother = cms.ESProducer("FlexibleKFFittingSmootherESProducer",
    ComponentName = cms.string('FlexibleKFFittingSmoother'),
    standardFitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    looperFitter = cms.string('LooperFittingSmoother'),
)
