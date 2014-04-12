import FWCore.ParameterSet.Config as cms

BeamSpotNominal = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
    ),
    timetype = cms.string('runtime'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('BeamSpotObjectsRcd'),
        tag = cms.string('NominalCollision')
    )),
    connect = cms.string('frontier://FrontierDev/CMS_COND_BEAMSPOT') ##FrontierDev/CMS_COND_BEAMSPOT"

)


