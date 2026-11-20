import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing("analysis")
options.register("dumpCells", False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool,
                 "write one line per cell")
options.parseArguments()

process = cms.Process("DIGEST")
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(options.inputFiles))
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.hgcalDigiDigest = cms.EDAnalyzer("HGCalDigiDigest",
                                         fileName=cms.untracked.string(options.outputFile),
                                         dumpCells=cms.untracked.bool(options.dumpCells))
process.p = cms.Path(process.hgcalDigiDigest)
