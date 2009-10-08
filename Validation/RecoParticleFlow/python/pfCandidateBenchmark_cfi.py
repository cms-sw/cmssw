import FWCore.ParameterSet.Config as cms


pfCandidateBenchmark = cms.EDAnalyzer("PFCandidateBenchmarkAnalyzer",
                                      OutputFile = cms.untracked.string('benchmark.root'),
                                      InputCollection = cms.InputTag('particleFlow'),
                                      BenchmarkLabel = cms.string('particleFlow')

                                      )
