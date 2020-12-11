import FWCore.ParameterSet.Config as cms

process = cms.Process("convertLHE2HepMC")
process.load("FWCore.MessageLogger.MessageLogger_cfi")    

process.MessageLogger.cerr.INFO = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.Generator = cms.untracked.PSet(limit = cms.untracked.int32(0))
process.MessageLogger.cerr.LHEInterface = cms.untracked.PSet(limit = cms.untracked.int32(0))
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10000)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)
process.source = cms.Source("LHESource",
    fileNames = cms.untracked.vstring()
)
process.source.fileNames = ([
'/store/eos/user/lenzip/Madgraph/8TeV/W1JetsToLNu_8TeV-madgraph/W1JetsToLNu_8TeV-madgraph_10001.lhe',
'/store/eos/user/lenzip/Madgraph/8TeV/W1JetsToLNu_8TeV-madgraph/W1JetsToLNu_8TeV-madgraph_10002.lhe',
'/store/eos/user/lenzip/Madgraph/8TeV/W1JetsToLNu_8TeV-madgraph/W1JetsToLNu_8TeV-madgraph_10003.lhe',
'/store/eos/user/lenzip/Madgraph/8TeV/W1JetsToLNu_8TeV-madgraph/W1JetsToLNu_8TeV-madgraph_10004.lhe',
'/store/eos/user/lenzip/Madgraph/8TeV/W1JetsToLNu_8TeV-madgraph/W1JetsToLNu_8TeV-madgraph_10005.lhe',
'/store/eos/user/lenzip/Madgraph/8TeV/W1JetsToLNu_8TeV-madgraph/W1JetsToLNu_8TeV-madgraph_10006.lhe',
'/store/eos/user/lenzip/Madgraph/8TeV/W1JetsToLNu_8TeV-madgraph/W1JetsToLNu_8TeV-madgraph_10007.lhe',
'/store/eos/user/lenzip/Madgraph/8TeV/W1JetsToLNu_8TeV-madgraph/W1JetsToLNu_8TeV-madgraph_10008.lhe',
'/store/eos/user/lenzip/Madgraph/8TeV/W1JetsToLNu_8TeV-madgraph/W1JetsToLNu_8TeV-madgraph_10009.lhe',
'/store/eos/user/lenzip/Madgraph/8TeV/W1JetsToLNu_8TeV-madgraph/W1JetsToLNu_8TeV-madgraph_10010.lhe'
])

process.load('Configuration.EventContent.EventContent_cff')                                                                                                   
process.load('Configuration/StandardSequences/EndOfProcess_cff')                                                                                              
process.load("GeneratorInterface.LHEInterface.lheCOMWeightProducer")
process.load('Configuration.StandardSequences.Generator_cff')
process.load("GeneratorInterface.LHEInterface.lhe2HepMCConverter_cfi")
process.genParticles.src = 'lhe2HepMCConverter'
process.load("Validation.EventGenerator.BasicGenValidation_cff")                                                                                            
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
ANALYSISEventContent = cms.PSet(
  outputCommands = cms.untracked.vstring('drop *')
)
ANALYSISEventContent.outputCommands.extend(process.MEtoEDMConverterFEVT.outputCommands)

process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('output8TeVReweight.root'),
  outputCommands = ANALYSISEventContent.outputCommands
)

# DQM Services

from DQMServices.Components.DQMEnvironment_cfi import *

DQMStore = cms.Service("DQMStore")

dqmSaver.convention = 'Offline'
dqmSaver.workflow = '/BasicHepMCValidation/Workflow/GEN'

from Validation.EventGenerator.genvalidTools import *

switchGenSourceForValidation(process, cms.InputTag('lhe2HepMCConverter'))
useExternalWeightForValidation(process, cms.VInputTag(cms.InputTag('lheCOMWeightProducer', 'comTo7000'))) 


process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('output8TeVReweight.root'),
  outputCommands = ANALYSISEventContent.outputCommands
)


process.gen = cms.Sequence(process.GeneInfo+process.genJetMET)
process.p = cms.Path( process.lheCOMWeightProducer +
                      process.lhe2HepMCConverter + 
                      process.gen +
                      process.genvalid_all + 
                      process.endOfProcess )
process.e = cms.EndPath(process.out)
