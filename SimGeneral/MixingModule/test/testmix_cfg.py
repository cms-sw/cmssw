import FWCore.ParameterSet.Config as cms

process = cms.Process("PRODTMIX")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/Cum.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.test = cms.EDProducer("TestMix",
    PrintLevel = cms.untracked.int32(2)
)

process.p = cms.Path(process.test)
