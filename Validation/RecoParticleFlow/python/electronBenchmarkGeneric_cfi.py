import FWCore.ParameterSet.Config as cms

electrons = 'put here the electron collection'

electronBenchmarkGeneric = cms.EDAnalyzer("GenericBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('electronsBenchmarkGeneric.root'),
    InputTruthLabel = cms.InputTag('put here the collection of generated electrons'),
    minEta = cms.double(-1),
    maxEta = cms.double(2.5),
    recPt = cms.double(10.0),
    deltaRMax = cms.double(0.2),
    StartFromGen = cms.bool(False),
    PlotAgainstRecoQuantities = cms.bool(False),
    OnlyTwoJets = cms.bool(False),
    BenchmarkLabel = cms.string( electrons ),
    InputRecoLabel = cms.InputTag( electrons )
)
