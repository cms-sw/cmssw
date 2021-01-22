import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
options.register('unitTest',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "are we running the unit test?")
options.register('inputFile',
                 "EarlyCollision.db", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "location of the input data")
options.parseArguments()

process = cms.Process("readDB")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("CondCore.CondDB.CondDB_cfi")

if(options.unitTest):
    #in case you want to read from sqlite file
    print(options.inputFile)
    process.CondDB.connect = cms.string('sqlite_file:'+options.inputFile)
else:
    process.CondDB.connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')

process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",process.CondDB,
                                        toGet = cms.VPSet(cms.PSet(
                                            record = cms.string('BeamSpotObjectsRcd'),
                                            # change the else clause in case with your favourite BeamSpot
                                            tag = cms.string('EarlyCollision') if options.unitTest else cms.string('BeamSpotObjects_Realistic25ns_13TeV2016Collisions_v1_mc'))
                                            )
                                        )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.beamspot = cms.EDAnalyzer("BeamSpotFromDB")
process.p = cms.Path(process.beamspot)

