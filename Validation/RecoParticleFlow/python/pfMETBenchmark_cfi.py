import FWCore.ParameterSet.Config as cms

pfMET = 'pfMet'

pfMETBenchmark = cms.EDAnalyzer("PFMETBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('METBenchmark.root'),
    InputTruthLabel = cms.InputTag('genParticles'),
    InputCaloLabel = cms.InputTag('met'),
    pfjBenchmarkDebug = cms.bool(False),                           
    PlotAgainstRecoQuantities = cms.bool(False),
    BenchmarkLabel = cms.string( pfMET ),
    InputRecoLabel = cms.InputTag( pfMET ),
    InputTCLabel = cms.InputTag( 'tcMet' )
)
