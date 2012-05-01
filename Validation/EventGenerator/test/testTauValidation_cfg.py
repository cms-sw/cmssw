import FWCore.ParameterSet.Config as cms

process = cms.Process("TauTest")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Validation.EventGenerator.TauValidation_cfi")
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(100)
    input = cms.untracked.int32(-1)    
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#	'file:gen.root'
#	'/store/relval/CMSSW_3_6_0_pre5/RelValHiggs200ChargedTaus/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0010/E09B3F44-E13D-DF11-995D-00304867BFA8.root'
	'rfio:/castor/cern.ch/user/s/slehti/HiggsAnalysisData/test_H120_100_1_08t_RAW_RECO.root'
    ) 
)

ANALYSISEventContent = cms.PSet(
#    outputCommands = cms.untracked.vstring('drop *')
    outputCommands = cms.untracked.vstring('keep *_*_*_TauTest')
)
#ANALYSISEventContent.outputCommands.extend(process.MEtoEDMConverterFEVT.outputCommands)

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

process.p = cms.Path(process.tauValidation+process.endOfProcess+process.out)




