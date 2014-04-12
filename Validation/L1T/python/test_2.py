import FWCore.ParameterSet.Config as cms

process = cms.Process("L1Val")

process.load('Configuration.StandardSequences.Services_cff')
process.load("Validation.L1T.L1Validator_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
#process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
#process.load('Configuration.EventContent.EventContent_cff')
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing("analysis")
options.parseArguments()

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring (options.inputFiles),
)

#process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
#    splitLevel = cms.untracked.int32(0),
#    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
#    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
#    fileName = cms.untracked.string('L1Validation.root'),
#    dataset = cms.untracked.PSet(
#        filterName = cms.untracked.string(''),
#        dataTier = cms.untracked.string('')
#    )
#)


process.val_step = cms.Path(process.L1Validator)
#process.endjob_step = cms.EndPath(process.endOfProcess)
#process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.dqmsave_step = cms.Path(process.DQMSaver)

process.schedule = cms.Schedule(process.val_step, process.dqmsave_step)
