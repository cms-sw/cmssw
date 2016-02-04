import FWCore.ParameterSet.Config as cms

process = cms.Process("write2DB")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CondCore.DBCommon.CondDBCommon_cfi")

#################################
# Produce a SQLITE FILE
#
process.CondDBCommon.connect = "SQLITEFILE"
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
#################################

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          toPut = cms.VPSet(cms.PSet(
    record = cms.string('BeamSpotObjectsRcd'),
    tag = cms.string('TAGNAME') )),
    timetype = cms.untracked.string('TIMETYPE'),                                     
    loadBlobStreamer = cms.untracked.bool(False)
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(1)
            )
process.beamspot = cms.EDAnalyzer("BeamSpotWrite2DB",
                                OutputFileName = cms.untracked.string('BEAMSPOTFILE')
                                )

process.p = cms.Path(process.beamspot)

# done.

