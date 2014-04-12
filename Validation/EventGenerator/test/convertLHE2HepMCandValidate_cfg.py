import FWCore.ParameterSet.Config as cms

process = cms.Process("convertLHE2HepMC")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("LHESource",
    fileNames = cms.untracked.vstring()
)
process.source.fileNames = ([
        '/store/lhe/3589/7TeV_wz2l2q_run50001_unweighted_events_qcut18_mgPostv2.lhe',
        '/store/lhe/3589/7TeV_wz2l2q_run50002_unweighted_events_qcut18_mgPostv2.lhe',
        '/store/lhe/3589/7TeV_wz2l2q_run50003_unweighted_events_qcut18_mgPostv2.lhe',
])

process.load("GeneratorInterface.LHEInterface.lhe2HepMCConverter_cfi")
process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")
process.genParticles.src = 'lhe2HepMCConverter'
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load("RecoJets.Configuration.RecoGenJets_cff")
process.load("RecoMET.Configuration.RecoGenMET_cff")
process.load("RecoJets.Configuration.GenJetParticles_cff")
process.load("RecoMET.Configuration.GenMETParticles_cff")


#process.load("Validation.EventGenerator.MBUEandQCDValidation_cff")                                                                                           
process.load("Validation.EventGenerator.BasicHepMCValidation_cff")                                                                                            
#process.mbueAndqcdValidation.hepmcCollection = 'lhe2HepMCConverter'                                                                                         
process.basicHepMCValidation.hepmcCollection = 'lhe2HepMCConverter'                                                                                 
process.load('Configuration.EventContent.EventContent_cff')                                                                                                   
process.load('Configuration/StandardSequences/EndOfProcess_cff')                                                                                              
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
ANALYSISEventContent = cms.PSet(
        outputCommands = cms.untracked.vstring('drop *')
        )
ANALYSISEventContent.outputCommands.extend(process.MEtoEDMConverterFEVT.outputCommands)

process.out = cms.OutputModule("PoolOutputModule",
                                   fileName = cms.untracked.string('output.root'),
                                   outputCommands = ANALYSISEventContent.outputCommands
                               )

# DQM Services

from DQMServices.Components.DQMEnvironment_cfi import *

DQMStore = cms.Service("DQMStore")

dqmSaver.convention = 'Offline'
dqmSaver.workflow = '/BasicHepMCValidation/Workflow/GEN'

process.p = cms.Path(process.lhe2HepMCConverter + 
                      process.genParticles +
                      process.genJetParticles + 
                      process.recoGenJets + 
                      process.genMETParticles +
                      process.recoGenMET
                      +process.basicHepMCValidation+process.endOfProcess
                      )
process.e = cms.EndPath(process.out)
