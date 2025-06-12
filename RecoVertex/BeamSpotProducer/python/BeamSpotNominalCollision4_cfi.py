import FWCore.ParameterSet.Config as cms

BeamSpotNominal4 = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
    ),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('BeamSpotObjectsRcd'),
        tag = cms.string('NominalCollision4')
    )),
    connect = cms.string('frontier://CoralDev/CMS_COND_BEAMSPOT') ##CoralDev/CMS_COND_BEAMSPOT"

)


