import FWCore.ParameterSet.Config as cms

from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
TransientTrackBuilderESProducer = cms.ESProducer("TransientTrackBuilderESProducer",
    ComponentName = cms.string('TransientTrackBuilder')
)


