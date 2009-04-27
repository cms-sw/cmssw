import FWCore.ParameterSet.Config as cms


pfElectronBenchmarkGeneric = cms.EDAnalyzer("GenericBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('electronBenchmarkGeneric.root'),
    InputTruthLabel = cms.InputTag(''),
    minEta = cms.double(-1),
    maxEta = cms.double(2.5),
    recPt = cms.double(10.0),
    deltaRMax = cms.double(0.2),
    StartFromGen = cms.bool(False),
    PlotAgainstRecoQuantities = cms.bool(False),
    OnlyTwoJets = cms.bool(False),
    BenchmarkLabel = cms.string( 'PFlowElectrons' ),
    InputRecoLabel = cms.InputTag('')
)
