import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource",
   firstRun = cms.untracked.uint32(1)
 )

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1)
)

process.analyzer = cms.EDAnalyzer("PluginWrapperKernel_uses_FunctionAnalyzer")

process.path = cms.Path( process.analyzer )
