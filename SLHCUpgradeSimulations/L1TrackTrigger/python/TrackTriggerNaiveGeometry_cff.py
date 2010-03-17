
import FWCore.ParameterSet.Config as cms

trackTriggerNaiveGeomRcdSrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('TrackTriggerNaiveGeometryRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

trackTriggerNaiveGeom = cms.ESProducer("TrackTriggerNaiveGeometryESProducer",
                                       radii = cms.vdouble( 100. ),
                                       lengths = cms.vdouble( 120. ),
                                       barrelTowZSize = cms.vdouble( 8.0 ),
                                       barrelTowPhiSize = cms.vdouble( 8.268 ),
                                       barrelPixelZSize = cms.vdouble( 1.0 ),
                                       barrelPixelPhiSize = cms.vdouble( 0.015 )
)                                       
                                     

