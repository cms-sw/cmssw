import FWCore.ParameterSet.Config as cms
process = cms.Process("readDB")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = cms.string('SQLITEFILE')
process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDB,
                                        toGet = cms.VPSet(cms.PSet(record = cms.string('BeamSpotObjectsRcd'),
                                                                   tag = cms.string('TAGNAME')
                                                                   )
                                                      )
                                        )

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.beamspot = cms.EDAnalyzer("BeamSpotFromDB")
process.p = cms.Path(process.beamspot)

