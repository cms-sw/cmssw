import FWCore.ParameterSet.Config as cms


pfTauBenchmarkGeneric = cms.EDAnalyzer("GenericBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('benchmark.root'),
    InputTruthLabel = cms.InputTag(''),
    minEta = cms.double(-1),
    maxEta = cms.double(2.8),
    recPt = cms.double(10.0),
    deltaRMax = cms.double(0.3),
    StartFromGen = cms.bool(True),
    PlotAgainstRecoQuantities = cms.bool(False),
    OnlyTwoJets = cms.bool(False),
    BenchmarkLabel = cms.string( 'PFlowTaus' ),
    InputRecoLabel = cms.InputTag( ''),                                   
    minDeltaEt = cms.double(-100.),
    maxDeltaEt = cms.double(50.),
    minDeltaPhi = cms.double(-0.5),
    maxDeltaPhi = cms.double(0.5),
    doMetPlots  = cms.bool(False)
)
