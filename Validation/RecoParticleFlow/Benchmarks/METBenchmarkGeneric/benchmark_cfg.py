# test file for PFCandidate validation
# performs a matching with the genParticles collection. 
# creates a root file with histograms filled with PFCandidate data,
# present in the Candidate, and in the PFCandidate classes, for matched
# PFCandidates. Matching histograms (delta pt etc) are also available. 

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

fa = 'RelValQCD'
fb = 'FlatPt_15_3000_Fast'
fc = 'ParticleFlow'

process.load("RecoParticleFlow.Configuration.DBS_Samples.%s_%s_cfi" % (fa, fb) )
#process.source = cms.Source("PoolSource",
#fileNames = cms.untracked.vstring(
#'/../user/l/lacroix/MET_Validation/ttbar_fastsim_310_pre6_muonAndJEC/aod.root'
#)
#)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("Validation.RecoParticleFlow.metBenchmark_cff")

process.dqmSaver.convention = 'Offline'
#process.dqmSaver.workflow = '/%s/%s/%s' % (fa, fb, fc)
process.dqmSaver.workflow = '/A/B/C'
process.dqmEnv.subSystemFolder = 'ParticleFlow'

process.p =cms.Path(
    process.dqmEnv +
    process.metBenchmarkSequence +
    process.dqmSaver
    )


process.schedule = cms.Schedule(process.p)


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100
