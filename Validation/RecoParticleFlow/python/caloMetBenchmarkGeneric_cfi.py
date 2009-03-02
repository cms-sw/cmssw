import FWCore.ParameterSet.Config as cms

met = 'put here the calo MET collection'

caloMetBenchmarkGeneric = cms.EDAnalyzer("GenericBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('metBenchmarkGeneric.root'),
    InputTruthLabel = cms.InputTag('put here the GenMET collection'),
    minEta = cms.double(-1),
    maxEta = cms.double(3.0),
    recPt = cms.double(10.0),
    deltaRMax = cms.double(999),
    PlotAgainstRecoQuantities = cms.bool(False),
    OnlyTwoJets = cms.bool(False),
    BenchmarkLabel = cms.string( met ),
    InputRecoLabel = cms.InputTag( met )
)
