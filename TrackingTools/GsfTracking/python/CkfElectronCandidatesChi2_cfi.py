import FWCore.ParameterSet.Config as cms

#
# Looser chi2 estimator for electron trajectory building
#   (definition should be moved?)
#
electronChi2 = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('electronChi2'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(100.0)
)


