import FWCore.ParameterSet.Config as cms

met = 'met'

caloMETBenchmarkGeneric = cms.EDAnalyzer("GenericBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('benchmark.root'),
    InputTruthLabel = cms.InputTag('genMetTrue'),
    minEta = cms.double(-5.0),
    maxEta = cms.double(5.0),
    recPt = cms.double(0.0),
    deltaRMax = cms.double(999),
    StartFromGen = cms.bool(True),
    PlotAgainstRecoQuantities = cms.bool(False),
    OnlyTwoJets = cms.bool(False),
    BenchmarkLabel = cms.string( met ),
    InputRecoLabel = cms.InputTag( met ),
    minDeltaEt = cms.double(-500.),
    maxDeltaEt = cms.double(500.),
    minDeltaPhi = cms.double(-3.2),
    maxDeltaPhi = cms.double(3.2),
    doMetPlots  = cms.bool(True)
)
