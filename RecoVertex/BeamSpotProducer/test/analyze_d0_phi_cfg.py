import FWCore.ParameterSet.Config as cms

process = cms.Process("d0phi")
# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1500)
)
process.p = cms.Path(process.d0_phi_analyzer)
process.MessageLogger.debugModules = ['BeamSpotAnalyzer']
process.d0_phi_analyzer.OutputFileName = 'EarlyCollision.root'


