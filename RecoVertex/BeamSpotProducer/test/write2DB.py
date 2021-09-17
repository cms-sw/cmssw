import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
options.register('unitTest',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "are we running the unit test?")
options.register('inputFile',
                 "EarlyCollision.txt", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "location of the input data")
options.parseArguments()

process = cms.Process("write2DB")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("CondCore.CondDB.CondDB_cfi")

#################################
# Produce a SQLITE FILE
process.CondDB.connect = "sqlite_file:EarlyCollision.db"
#################################

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('BeamSpotObjectsRcd'),
                                                                     tag = cms.string('EarlyCollision'))),
                                          loadBlobStreamer = cms.untracked.bool(False)
                                          )

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.beamspot = cms.EDAnalyzer("BeamSpotWrite2DB",
                                  OutputFileName = cms.untracked.string(options.inputFile)
                                  )

process.p = cms.Path(process.beamspot)
