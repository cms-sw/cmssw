import FWCore.ParameterSet.Config as cms

BeamSpotNominal1 = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
    ),
    timetype = cms.string('runtime'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('BeamSpotObjectsRcd'),
        tag = cms.string('NominalCollision1')
    )),
    connect = cms.string('frontier://CoralDev/CMS_COND_BEAMSPOT') ##CoralDev/CMS_COND_BEAMSPOT"

)


