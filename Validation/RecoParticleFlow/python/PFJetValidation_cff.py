import FWCore.ParameterSet.Config as cms

from DQMOffline.PFTau.PFJetDQMAnalyzer_cfi import pfJetDQMAnalyzer

pfJetValidation1 = pfJetDQMAnalyzer.clone(
    InputCollection = 'ak4PFJets',
    MatchCollection = 'ak4GenJets',
    BenchmarkLabel  = 'PFJetValidation/CompWithGenJet'
)

pfJetValidation2 = pfJetDQMAnalyzer.clone(
    InputCollection = 'ak4PFJets',
    MatchCollection = 'ak4CaloJets',
    BenchmarkLabel  = 'PFJetValidation/CompWithCaloJet'
)

pfJetValidationSequence = cms.Sequence( pfJetValidation1 * pfJetValidation2 )
