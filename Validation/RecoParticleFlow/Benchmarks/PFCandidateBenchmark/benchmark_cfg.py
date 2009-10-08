# Runs the GenericBenchmark to
# monitor performance of pfMET

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")


process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(
'/../user/l/lacroix/MET_Validation/ttbar_fastsim_310_pre6_muonAndJEC/aod.root'
)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.load("Validation.RecoParticleFlow.pfCandidateBenchmark_cff")

process.p =cms.Path(
    process.pfCandidateBenchmarkSequence
    )


process.schedule = cms.Schedule(process.p)



process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 50
