import FWCore.ParameterSet.Config as cms


process = cms.Process("PRODMIX")
process.load("SimGeneral.DataMixingModule.mixOne_sim_on_sim_cfi")

process.load("Configuration.EventContent.EventContent_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger = cms.Service("MessageLogger",
     destinations = cms.untracked.vstring('cout'),
     cout = cms.untracked.PSet( threshold = cms.untracked.string("DEBUG") ),
     debugModules = cms.untracked.vstring('DataMixingModule')
)


process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        mix = cms.untracked.uint32(12345)
    )
)

process.Tracer = cms.Service("Tracer",
    indention = cms.untracked.string('$$')
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0003/0A0241FB-9182-DD11-98E1-001617E30D40.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)
process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('file:MixTest.root')
)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path(process.mix+process.dump)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.p,process.outpath)



