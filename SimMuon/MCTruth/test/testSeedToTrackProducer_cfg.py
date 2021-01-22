import FWCore.ParameterSet.Config as cms


process = cms.Process("test")

 # MessageLogger
process.load("FWCore.MessageService.MessageLogger_cfi")

'''process.MessageLogger.SeedToTrackProducer = dict()
process.MessageLogger.cout = cms.untracked.PSet(
                                                noTimeStamps = cms.untracked.bool(True),
                                                threshold = cms.untracked.string('INFO'),
                                                INFO = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                                default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                                )
'''
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag.globaltag = 'START70_V2::All'


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                            '/store/relval/CMSSW_7_0_0_pre9/RelValZMM/GEN-SIM-RECO/START70_V2-v4/00000/CAD227AF-9E5D-E311-B8EE-0025905A608E.root'
                                                              )
                            )


process.myProducerLabel = cms.EDProducer('SeedToTrackProducer',
                L2seedsCollection = cms.InputTag("ancientMuonSeed")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testOutput.root'),
    outputCommands = cms.untracked.vstring('drop *'
                                           ,'keep *_*_*_test'  )
)

  
process.p = cms.Path(process.myProducerLabel)

process.e = cms.EndPath(process.out)
