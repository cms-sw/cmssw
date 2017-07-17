import FWCore.ParameterSet.Config as cms

jets = 'iterativeCone5CaloJets'

caloJetBenchmarkGeneric = cms.EDAnalyzer("GenericBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('jetBenchmarkGeneric.root'),
    InputTruthLabel = cms.InputTag('ak4GenJets'),
    minEta = cms.double(-1),
    maxEta = cms.double(1.4),
    recPt = cms.double(10.0),
    deltaRMax = cms.double(0.2),
    StartFromGen = cms.bool(False),
    PlotAgainstRecoQuantities = cms.bool(False),
    OnlyTwoJets = cms.bool(True),
    BenchmarkLabel = cms.string( jets ),
    InputRecoLabel = cms.InputTag( jets ),                                   
    minDeltaEt = cms.double(-100.),
    maxDeltaEt = cms.double(50.),
    minDeltaPhi = cms.double(-0.5),
    maxDeltaPhi = cms.double(0.5),
    doMetPlots  = cms.bool(False)
)
