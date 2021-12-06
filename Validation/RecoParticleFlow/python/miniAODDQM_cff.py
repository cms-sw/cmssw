import FWCore.ParameterSet.Config as cms

# jet
from DQMOffline.PFTau.PFJetDQMAnalyzer_cfi import pfJetDQMAnalyzer

slimmedJetValidation1 = pfJetDQMAnalyzer.clone(
    BenchmarkLabel  = 'slimmedJetValidation/CompWithPFJets',
    InputCollection = 'slimmedJets',
    MatchCollection = 'ak4PFJetsCHS', # ak5PFJetsCHS # ak5PFJets
    ptMin = 10.0,
    CreatePFractionHistos = True
    #InputCollection = 'ak5PFJets'
    #MatchCollection = 'slimmedJets'
)

slimmedJetValidation2 = pfJetDQMAnalyzer.clone(
    BenchmarkLabel  = 'slimmedJetValidation/CompWithPFJetsEC',
    #InputCollection = JetValidation1.MatchCollection
    #MatchCollection = JetValidation1.InputCollection
    InputCollection = 'slimmedJets',
    MatchCollection = 'ak4PFJetsNewL1Fast23', # ak4PFJetsCHSEC # ak4PFJetsCHS
    ptMin = slimmedJetValidation1.ptMin,
    CreatePFractionHistos = True
)


# jetRes plots
from DQMOffline.PFTau.PFJetResDQMAnalyzer_cfi import pfJetResDQMAnalyzer

slimmedJetResValidation1 = pfJetResDQMAnalyzer.clone(
    InputCollection = slimmedJetValidation1.InputCollection,
    MatchCollection = slimmedJetValidation1.MatchCollection,
    ptMin = slimmedJetValidation1.ptMin
)

slimmedJetResValidation2 = pfJetResDQMAnalyzer.clone(
    InputCollection = slimmedJetValidation2.InputCollection,
    MatchCollection = slimmedJetValidation2.MatchCollection,
    ptMin = slimmedJetValidation2.ptMin
)


# MET
from DQMOffline.PFTau.PFMETDQMAnalyzer_cfi import pfMETDQMAnalyzer

slimmedMETValidation1 = pfMETDQMAnalyzer.clone(
    BenchmarkLabel  = 'slimmedMETValidation/CompWithPFMET',
    InputCollection = 'slimmedMETs',
    MatchCollection = 'pfMet'
)

slimmedMETValidation2 = pfMETDQMAnalyzer.clone(
    BenchmarkLabel  = 'slimmedMETValidation/CompWithPFMETT1',
    InputCollection = 'slimmedMETs',
    MatchCollection = 'pfMetT1'
)


# muons
from DQMOffline.PFTau.PFMuonDQMAnalyzer_cfi import pfMuonDQMAnalyzer

slimmedMuonValidation1 = pfMuonDQMAnalyzer.clone(
    BenchmarkLabel  = 'slimmedMuonValidation/CompWithRecoMuons',
    InputCollection = 'slimmedMuons',
    MatchCollection = 'muons'
)


# electrons
from DQMOffline.PFTau.PFElectronDQMAnalyzer_cfi import pfElectronDQMAnalyzer

slimmedElectronValidation1 = pfElectronDQMAnalyzer.clone(
    BenchmarkLabel  = 'slimmedElectronValidation/CompWithGedGsfElectrons',
    InputCollection = 'slimmedElectrons',
    MatchCollection = 'gedGsfElectrons'
)


from JetMETCorrections.Type1MET.pfMETCorrectionType0_cfi import type0PFMEtCorrectionPFCandToVertexAssociationForValidationMiniAOD

miniAODDQMSequence = cms.Sequence(
    type0PFMEtCorrectionPFCandToVertexAssociationForValidationMiniAOD *
    slimmedJetValidation1 * slimmedJetValidation2 *
    slimmedJetResValidation1 * slimmedJetResValidation2 *
    slimmedMETValidation1 * slimmedMETValidation2 *
    slimmedMuonValidation1 *
    slimmedElectronValidation1
)
