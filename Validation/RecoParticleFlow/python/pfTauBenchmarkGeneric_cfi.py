import FWCore.ParameterSet.Config as cms

jets = 'iterativeCone5PFJets'
genjets = 'genJetSelector'

pfTauBenchmarkGeneric = cms.EDAnalyzer("GenericBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('benchmark.root'),
    InputTruthLabel = cms.InputTag( genjets ),
    minEta = cms.double(-1),
    maxEta = cms.double(2.8),
    recPt = cms.double(10.0),
    deltaRMax = cms.double(0.3),
    StartFromGen = cms.bool(True),
    PlotAgainstRecoQuantities = cms.bool(False),
    OnlyTwoJets = cms.bool(False),
    BenchmarkLabel = cms.string( jets ),
    InputRecoLabel = cms.InputTag( jets )
)
