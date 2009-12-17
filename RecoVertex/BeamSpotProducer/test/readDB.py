import FWCore.ParameterSet.Config as cms


process = cms.Process("write2DB")
process.load("FWCore.MessageLogger.MessageLogger_cfi")


process.load("CondCore.DBCommon.CondDBSetup_cfi")



process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDBSetup,
                                        toGet = cms.VPSet(cms.PSet(
    record = cms.string('BeamSpotObjectsRcd'),
    tag = cms.string('Early900GeVCollision_7p4cm_V1_IDEAL_V10')
    )),
                                        connect = cms.string('sqlite_file:EarlyCollision.db')
                                        #connect = cms.string('oracle://cms_orcon_prod/CMS_COND_21X_BEAMSPOT')
                                        )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(2)
                    )
process.beamspot = cms.EDFilter("BeamSpotFromDB")


process.p = cms.Path(process.beamspot)

