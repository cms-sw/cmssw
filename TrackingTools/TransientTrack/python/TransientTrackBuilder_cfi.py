import FWCore.ParameterSet.Config as cms

from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
from Geometry.CMSCommonData.cmsIdealGeometryXML_cff import *
TransientTrackBuilderESProducer = cms.ESProducer("TransientTrackBuilderESProducer",
    ComponentName = cms.string('TransientTrackBuilder')
)


