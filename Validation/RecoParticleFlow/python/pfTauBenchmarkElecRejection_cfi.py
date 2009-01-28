import FWCore.ParameterSet.Config as cms

pfTauBenchmarkElecRejection = cms.EDAnalyzer("PFTauElecRejectionBenchmarkAnalyzer",
    OutputFile = cms.untracked.string('tauBenchmarkElecRejection.root'),
    InputTruthLabel = cms.InputTag('source'),
    BenchmarkLabel = cms.string('PFTauElecRejection'), 
    minRecoPt = cms.double(15.0),
    maxRecoAbsEta = cms.double(2.5),
    minMCPt = cms.double(10.0),
    maxMCAbsEta = cms.double(2.5),
    maxDeltaR = cms.double(0.3),
    PFTauProducer = cms.InputTag('pfRecoTauProducerHighEfficiency'), 
    PFTauDiscriminatorByIsolationProducer = cms.InputTag('pfRecoTauDiscriminationByIsolationHighEfficiency'),
    PFTauDiscriminatorAgainstElectronProducer = cms.InputTag('pfRecoTauDiscriminationAgainstElectronHighEfficiency'),
    ApplyEcalCrackCut = cms.bool(True),
    GenMatchObjectLabel = cms.string('tau') #  match with hadronic 'tau' or electron "e"
)
