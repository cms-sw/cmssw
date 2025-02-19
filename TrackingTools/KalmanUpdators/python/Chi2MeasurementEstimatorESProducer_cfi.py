import FWCore.ParameterSet.Config as cms

Chi2MeasurementEstimator = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(30.0)
)


