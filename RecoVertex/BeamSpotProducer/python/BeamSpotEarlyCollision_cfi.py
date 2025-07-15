import FWCore.ParameterSet.Config as cms

BeamSpotEarlyCollision = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
    ),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('BeamSpotObjectsRcd'),
        tag = cms.string('EarlyCollision_5p3cm_mc')
    )),
    connect = cms.string('frontier://FrontierDev/CMS_COND_BEAMSPOT') ##FrontierDev/CMS_COND_BEAMSPOT"

)


