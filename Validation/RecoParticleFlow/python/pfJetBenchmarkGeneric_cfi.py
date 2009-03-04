import FWCore.ParameterSet.Config as cms

jets = 'iterativeCone5PFJets'

pfJetBenchmarkGeneric = cms.EDAnalyzer("GenericBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('jetBenchmarkGeneric.root'),
    InputTruthLabel = cms.InputTag('iterativeCone5GenJets'),
    minEta = cms.double(-1),
    maxEta = cms.double(1.4),
    recPt = cms.double(10.0),
    deltaRMax = cms.double(0.1),
    StartFromGen = cms.bool(False),
    PlotAgainstRecoQuantities = cms.bool(False),
    OnlyTwoJets = cms.bool(True),
    BenchmarkLabel = cms.string( jets ),
    InputRecoLabel = cms.InputTag( jets )
)
