import FWCore.ParameterSet.Config as cms

jets = 'iterativeCone5CaloJets'

caloTauBenchmarkGeneric = cms.EDAnalyzer("GenericBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('benchmark.root'),
    InputTruthLabel = cms.InputTag('tauGenJets'),
    minEta = cms.double(-1),
    maxEta = cms.double(2.8),
    recPt = cms.double(0.0),
    deltaRMax = cms.double(0.3),
    PlotAgainstRecoQuantities = cms.bool(False),
    OnlyTwoJets = cms.bool(False),
    BenchmarkLabel = cms.string( jets ),
    InputRecoLabel = cms.InputTag( jets )
)
