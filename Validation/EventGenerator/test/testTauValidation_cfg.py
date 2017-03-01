import FWCore.ParameterSet.Config as cms

process = cms.Process("TauTest")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.load("Validation.EventGenerator.TauValidation_cfi")
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.StandardSequences.Validation_cff')

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(100)
    input = cms.untracked.int32(-1)    
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	'file:/afs/cern.ch/user/i/inugent/tmp/CMSSW_7_4_X_2015-02-03-0200/src/00A47D7F-B88D-E411-A4A1-0025905B85D0.root'
    ) 
)

ANALYSISEventContent = cms.PSet(
#    outputCommands = cms.untracked.vstring('drop *')
    outputCommands = cms.untracked.vstring('keep *')
)
#ANALYSISEventContent.outputCommands.extend(process.MEtoEDMConverterFEVT.outputCommands)

#process.out = cms.OutputModule("PoolOutputModule",
#    verbose = cms.untracked.bool(False),
#    fileName = cms.untracked.string('output.root'),
#    outputCommands = ANALYSISEventContent.outputCommands
#)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step1_inDQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)


process.debugOutput = cms.OutputModule("PoolOutputModule",
                                       fileName = cms.untracked.string("Test_DYJetsToLL_M-50_TuneCUETP8M1_8TeV-amcatnloFXFX-pythia8.root"),
                                       outputCommands = cms.untracked.vstring('keep *'),
                                       )
process.out_step = cms.EndPath(process.debugOutput)


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

#process.schedule = cms.Schedule(process.tauValidation+process.endOfProcess+process.out_step)

#GenEventInfoProduct                   "generator"                 ""                "SIM"
process.tauValidation = cms.EDAnalyzer("TauValidation",
                                       genparticleCollection = cms.InputTag("genParticles",""),
                                       tauEtCutForRtau = cms.double(50),
                                       UseWeightFromHepMC = cms.bool(False)
                                       )

process.validation_step = cms.EndPath(process.tauValidation_seq)
process.out_step = cms.EndPath(process.debugOutput)
#process.validation_step = cms.EndPath(process.tauValidation)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
#process.schedule = cms.Schedule(process.validation_step)
#process.schedule.append(process.out_step)

process.schedule = cms.Schedule(process.validation_step,process.DQMoutput_step)
