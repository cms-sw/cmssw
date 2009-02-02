import FWCore.ParameterSet.Config as cms

met = 'pfMet'

pfMetBenchmarkGeneric = cms.EDAnalyzer("PFMETBenchmarkAnalyzer",
    InputTruthLabel = cms.InputTag('genParticles'),
    InputRecoLabel = cms.InputTag( met ),
    InputCaloLabel = cms.InputTag( 'met' ),
    OutputFile = cms.untracked.string('metBenchmarkGeneric.root'),
    pfjBenchmarkDebug = cms.bool(False),
    PlotAgainstRecoQuantities = cms.bool(False),
    BenchmarkLabel = cms.string( met )
#    maxEta = cms.double(3.0),
#    recPt = cms.double(10.0),
#    deltaRMax = cms.double(999),
#    OnlyTwoJets = cms.bool(False),
)

