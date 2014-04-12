import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("BasicGenTest")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Validation.EventGenerator.WValidation_cfi")
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.workflow = "/"+os.environ["CMSSW_VERSION"]+"/RelVal/Validation"
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(100)
    input = cms.untracked.int32(10000)    
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:Wenu_test.root') 
    #fileNames = cms.untracked.vstring('file:gen_sherpa.root') 
)

ANALYSISEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
ANALYSISEventContent.outputCommands.extend(process.MEtoEDMConverterFEVT.outputCommands)

process.out = cms.OutputModule("PoolOutputModule",
    verbose = cms.untracked.bool(False),
    fileName = cms.untracked.string('output.root'),
    outputCommands = ANALYSISEventContent.outputCommands
)

#Adding SimpleMemoryCheck service:
process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
    ignoreTotal=cms.untracked.int32(1),
    oncePerEventMode=cms.untracked.bool(False))
#Adding Timing service:
process.Timing=cms.Service("Timing",
    summaryOnly=cms.untracked.bool(True))

#Add these 3 lines to put back the summary for timing information at the end of the logfile
#(needed for TimeReport report)
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)    

process.p = cms.Path(process.wValidation+process.endOfProcess+process.out)
process.EDMtoMEconv_and_saver= cms.Path(process.EDMtoMEConverter*process.dqmSaver)



