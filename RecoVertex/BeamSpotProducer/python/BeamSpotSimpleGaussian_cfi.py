import FWCore.ParameterSet.Config as cms

BeamSpotGaussian = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
    ),
    timetype = cms.string('runtime'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('BeamSpotObjectsRcd'),
        tag = cms.string('SimpleGaussian')
    )),
    connect = cms.string('frontier://FrontierDev/CMS_COND_BEAMSPOT') ##FrontierDev/CMS_COND_BEAMSPOT"

)


