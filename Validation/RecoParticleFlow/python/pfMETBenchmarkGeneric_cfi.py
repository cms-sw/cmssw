import FWCore.ParameterSet.Config as cms

MET = 'pfMet'

pfMETBenchmarkGeneric = cms.EDAnalyzer("GenericBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('metBenchmarkGeneric.root'),
    InputTruthLabel = cms.InputTag('genParticles'),
    InputCaloLabel = cms.InputTag('met'),
    StartFromGen = cms.bool(False),
    PlotAgainstRecoQuantities = cms.bool(False),
    BenchmarkLabel = cms.string( MET ),
    InputRecoLabel = cms.InputTag( MET )
)
