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

#process.load("RecoParticleFlow.Configuration.DBS_Samples.%s_%s_cfi" % (fa, fb) )
process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(
#'/../user/l/lacroix/MET_Validation/ttbar_fastsim_310_pre6_muonAndJEC/aod.root'
#'/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/133/532/EC93873A-D74B-DF11-A1B9-00E08179185D.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/133/531/D6E1CE68-ED4B-DF11-A676-003048D45F84.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/133/529/223C34BD-EC4B-DF11-9CA6-003048D476D4.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/133/526/B28AEED6-E94B-DF11-9124-00E08178C103.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/133/521/C0EFDC20-024C-DF11-A82A-00E08178C155.root',
#        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/133/518/C620697E-B94B-DF11-A125-003048D46090.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/133/516/CA61DFDB-E14B-DF11-9193-003048D476B0.root',
)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.load("Validation.RecoParticleFlow.metBenchmark_cff")
process.pfMetBenchmark.mode = 1
process.caloMetBenchmark.mode = 1
process.UncorrCaloMetBenchmark.mode = 1

process.dqmSaver.convention = 'Offline'
#process.dqmSaver.workflow = '/%s/%s/%s' % (fa, fb, fc)
process.dqmSaver.workflow = '/A/B/C'
process.dqmEnv.subSystemFolder = 'ParticleFlow'

process.p =cms.Path(
    process.dqmEnv +
    process.metBenchmarkSequenceData +
    process.dqmSaver
    )


process.schedule = cms.Schedule(process.p)


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100
