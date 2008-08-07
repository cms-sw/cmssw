import FWCore.ParameterSet.Config as cms

TrackAssociatorByDeltaR1 = cms.ESProducer("TrackAssociatorByPositionESProducer",
    QminCut = cms.double(120.0),
    MinIfNoMatch = cms.bool(False),
    ComponentName = cms.string('TrackAssociatorByDeltaR1'),
    propagator = cms.string('SteppingHelixPropagatorAny'),
    positionMinimumDistance = cms.double(0.0),
    method = cms.string('posdr'),
    QCut = cms.double(0.5)
)


