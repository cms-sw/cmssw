import FWCore.ParameterSet.Config as cms

from DQMOffline.PFTau.PFJetResDQMAnalyzer_cfi import pfJetResDQMAnalyzer

pfJetResValidation1 = pfJetResDQMAnalyzer.clone(
    InputCollection = 'ak4PFJets', # for global Validation
    MatchCollection = 'ak4GenJets', # for global Validatio
    BenchmarkLabel  = 'PFJetResValidation/JetPtRes'
)
pfJetResValidationSequence = cms.Sequence( pfJetResValidation1 )

# NoTracking
pfJetResValidation2 = pfJetResDQMAnalyzer.clone(
    InputCollection = 'ak4PFJets::PFlowDQMnoTracking',
    MatchCollection = 'ak4GenJets::PFlowDQMnoTracking',
    BenchmarkLabel  = 'PFJetResValidation/JetPtResNoTracking'
)
pfJetResValidationSequence_NoTracking = cms.Sequence( pfJetResValidation2 )
