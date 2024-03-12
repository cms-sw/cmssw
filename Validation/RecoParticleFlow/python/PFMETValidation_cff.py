import FWCore.ParameterSet.Config as cms

from DQMOffline.PFTau.PFMETDQMAnalyzer_cfi import pfMETDQMAnalyzer

pfMETValidation1 = pfMETDQMAnalyzer.clone(
    InputCollection = 'pfMet',
    MatchCollection = 'genMetTrue',
    BenchmarkLabel  = 'PFMETValidation/CompWithGenMET'
)

pfMETValidation2 = pfMETDQMAnalyzer.clone(
    InputCollection = 'pfMet',
    MatchCollection = 'caloMet',
    BenchmarkLabel  = 'PFMETValidation/CompWithCaloMET'
)

pfMETValidationSequence = cms.Sequence( pfMETValidation1 * pfMETValidation2 )
# foo bar baz
# 0Lw9rKBCEuJ9A
