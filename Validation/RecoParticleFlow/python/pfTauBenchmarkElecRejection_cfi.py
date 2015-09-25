import FWCore.ParameterSet.Config as cms

pfTauBenchmarkElecRejection = cms.EDAnalyzer("PFTauElecRejectionBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('tauBenchmarkElecRejection.root'),
    InputTruthLabel = cms.InputTag('generatorSmeared'),
    BenchmarkLabel = cms.string('PFTauElecRejection'), 
    minRecoPt = cms.double(15.0),
    maxRecoAbsEta = cms.double(2.5),
    minMCPt = cms.double(10.0),
    maxMCAbsEta = cms.double(2.5),
    maxDeltaR = cms.double(0.3),
    PFTauProducer = cms.InputTag('shrinkingConePFTauProducer'), 
    PFTauDiscriminatorByIsolationProducer = cms.InputTag('shrinkingConePFTauDiscriminationByIsolation'),
    PFTauDiscriminatorAgainstElectronProducer = cms.InputTag('shrinkingConePFTauDiscriminationAgainstElectron'),
    ApplyEcalCrackCut = cms.bool(True),
    GenMatchObjectLabel = cms.string('tau') #  match with hadronic 'tau' or electron "e"
)
