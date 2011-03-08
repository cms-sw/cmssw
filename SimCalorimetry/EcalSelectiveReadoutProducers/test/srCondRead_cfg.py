import FWCore.ParameterSet.Config as cms

#tag = 'EcalSRSettings_v00_lowlumi_mc'
tag = 'EcalSRSettings_v00_beam10_mc'

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:' + tag + '.db'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

process.MessageLogger = cms.Service("MessageLogger",
                                      debugModules = cms.untracked.vstring('*'),
                                      destinations = cms.untracked.vstring('cout')
                                    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                      process.CondDBCommon,
                                      toGet = cms.VPSet(cms.PSet(
    record = cms.string('EcalSRSettingsRcd'),
    tag = cms.string(tag)
    )))


process.readFromDB = cms.EDAnalyzer("EcalSRCondTools",
    mode = cms.string("read")
)

process.p = cms.Path(process.readFromDB)

