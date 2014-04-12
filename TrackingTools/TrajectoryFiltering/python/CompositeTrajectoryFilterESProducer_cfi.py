import FWCore.ParameterSet.Config as cms

compositeTrajectoryFilterESProducer = cms.ESProducer("CompositeTrajectoryFilterESProducer",
    ComponentName = cms.string('compositeTrajectoryFilter'),
    filterNames = cms.vstring()
)


