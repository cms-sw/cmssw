import FWCore.ParameterSet.Config as cms

jets = 'iterativeCone5CaloJets'

caloTauBenchmarkGeneric = cms.EDAnalyzer("GenericBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('benchmark.root'),
    InputTruthLabel = cms.InputTag('tauGenJets'),
    maxEta = cms.double(1.0),
    recPt = cms.double(10.0),
    deltaRMax = cms.double(0.1),
    PlotAgainstRecoQuantities = cms.bool(False),
    OnlyTwoJets = cms.bool(False),
    BenchmarkLabel = cms.string( jets ),
    InputRecoLabel = cms.InputTag( jets )
)
