import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing("analysis")
options.register(
    "outTag", "All", options.multiplicity.singleton, options.varType.string, "outTag"
)
options.parseArguments()

process = cms.Process("HARVESTING")

process.source = cms.Source(
    "DQMRootSource", fileNames=cms.untracked.vstring(options.inputFiles)
)


process.load("DQMServices.Core.DQMStore_cfi")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(5000),
    limit = cms.untracked.int32(10000000)
)


from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
process.validationClient = DQMEDHarvester("HLTGenValClient",
    outputFileName = cms.untracked.string(''),
    subDirs        = cms.untracked.vstring("HLTGenVal"),
)


process.load("DQMServices.Components.DQMFileSaver_cfi")
process.dqmSaver.workflow = "/HLT/Validation/{}".format(options.outTag)

process.DQMFileSaverOutput = cms.EndPath(
    process.validationClient + process.dqmSaver
)
