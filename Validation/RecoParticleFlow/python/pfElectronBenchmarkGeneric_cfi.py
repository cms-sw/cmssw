import FWCore.ParameterSet.Config as cms


pfElectronBenchmarkGeneric = cms.EDAnalyzer("GenericBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('benchmark.root'),
    InputTruthLabel = cms.InputTag(''),
    minEta = cms.double(-1),
    maxEta = cms.double(2.5),
    recPt = cms.double(2.0),
    deltaRMax = cms.double(0.2),
    StartFromGen = cms.bool(False),
    PlotAgainstRecoQuantities = cms.bool(False),
    OnlyTwoJets = cms.bool(False),
    BenchmarkLabel = cms.string( 'PFlowElectrons' ),
    InputRecoLabel = cms.InputTag(''),
    minDeltaEt = cms.double(-100.),
    maxDeltaEt = cms.double(50.),
    minDeltaPhi = cms.double(-0.5),
    maxDeltaPhi = cms.double(0.5),
    doMetPlots  = cms.bool(False)
)
