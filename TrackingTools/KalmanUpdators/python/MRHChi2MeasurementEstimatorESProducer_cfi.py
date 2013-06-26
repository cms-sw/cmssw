import FWCore.ParameterSet.Config as cms

MRHChi2MeasurementEstimator = cms.ESProducer("MRHChi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('MRHChi2'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(30.0)
)


