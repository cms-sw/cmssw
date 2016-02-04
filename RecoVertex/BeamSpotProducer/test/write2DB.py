import FWCore.ParameterSet.Config as cms


process = cms.Process("write2DB")
process.load("FWCore.MessageLogger.MessageLogger_cfi")


process.load("CondCore.DBCommon.CondDBCommon_cfi")

#################################
# Produce a SQLITE FILE
#
process.CondDBCommon.connect = "sqlite_file:Early900GeVCollision_7p4cm_STARTUP_mc.db"
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
#################################
#
# upload conditions to orcon
#
#process.CondDBCommon.connect = "oracle://cms_orcoff_prep/CMS_COND_BEAMSPOT"
#process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = 'MC_31X_V2::All'

#################################

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          toPut = cms.VPSet(cms.PSet(
    record = cms.string('BeamSpotObjectsRcd'),
    tag = cms.string('Early900GeVCollision_7p4cm_STARTUP_mc') )),
    loadBlobStreamer = cms.untracked.bool(False)
)



process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(1)
                    )
process.beamspot = cms.EDAnalyzer("BeamSpotWrite2DB",
                                OutputFileName = cms.untracked.string('BFMC900GeVstartup.txt')
                                )



#process.CondDBCommon.connect = 'oracle://cms_orcoff_int2r/CMS_COND_BEAMSPOT'
#process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

process.p = cms.Path(process.beamspot)

