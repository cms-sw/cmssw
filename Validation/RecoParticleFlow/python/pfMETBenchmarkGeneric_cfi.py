import FWCore.ParameterSet.Config as cms

met = 'pfMet'

pfMETBenchmarkGeneric = cms.EDAnalyzer("GenericBenchmarkAnalyzer",
    InputTruthLabel = cms.InputTag('genMetTrue'),
    InputRecoLabel = cms.InputTag( met ),
    InputCaloLabel = cms.InputTag( 'met' ),
    InputTCLabel = cms.InputTag( 'tcMet' ),    
    OutputFile = cms.untracked.string('benchmark.root'),
    pfjBenchmarkDebug = cms.bool(False),
    PlotAgainstRecoQuantities = cms.bool(False),
    BenchmarkLabel = cms.string( met ),
    minEta = cms.double(-5.0),                       
    maxEta = cms.double(5.0),
    recPt = cms.double(0.0),
    deltaRMax = cms.double(999),
    StartFromGen = cms.bool(True),
    OnlyTwoJets = cms.bool(False),
    minDeltaEt = cms.double(-500.),
    maxDeltaEt = cms.double(500.),
    minDeltaPhi = cms.double(-3.2),
    maxDeltaPhi = cms.double(3.2),
    doMetPlots  = cms.bool(True)
)
