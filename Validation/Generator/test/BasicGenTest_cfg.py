import FWCore.ParameterSet.Config as cms

process = cms.Process('Validation')

process.load("Validation.RecoB.BasicGenTest_cff")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.Validation = cms.EDFilter("BasicGenTest")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring ('file:/uscms/home/jaz8600/CMSSW_3_1_0_pre11/src/Validation/RecoB/data/TTbar.root'))


process.p1 = cms.Path(process.Validation*process.dqmSaver)
