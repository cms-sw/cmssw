import FWCore.ParameterSet.Config as cms

#pfJets = 'iterativeCone5PFJets'
pfJets = 'ak5PFJets'

pfJetBenchmark = cms.EDAnalyzer("PFJetBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('JetBenchmark.root'),
    InputTruthLabel = cms.InputTag('ak5GenJets'),
    maxEta = cms.double(5.0),
    recPt = cms.double(10.0),
    pfjBenchmarkDebug = cms.bool(False),                           
    deltaRMax = cms.double(0.1),
    PlotAgainstRecoQuantities = cms.bool(False),
    OnlyTwoJets = cms.bool(True),
    BenchmarkLabel = cms.string( pfJets ),
    InputRecoLabel = cms.InputTag( pfJets )
)
