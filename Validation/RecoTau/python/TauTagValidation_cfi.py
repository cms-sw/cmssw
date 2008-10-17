# This test config file is used to validate the pfRecoTauProducer
# 

import FWCore.ParameterSet.Config as cms

PFTausHighEfficiencyBothProngs = cms.EDAnalyzer("TauTagValidation",
   DataType                     = cms.string('Leptons'),               
   MatchDeltaR_Leptons          = cms.double(0.15),
   MatchDeltaR_Jets             = cms.double(0.3),
   SaveOutputHistograms         = cms.bool(True),
   RefCollection                = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
   ExtensionName                = cms.string("ALLTHREE"),
   TauProducer                  = cms.string('pfRecoTauProducerHighEfficiency'),
   TauProducerDiscriminators    = cms.untracked.vstring("pfRecoTauDiscriminationByLeadingTrackFindingHighEfficiency","pfRecoTauDiscriminationByLeadingTrackPtCutHighEfficiency","pfRecoTauDiscriminationByIsolationHighEfficiency","pfRecoTauDiscriminationAgainstElectronHighEfficiency","pfRecoTauDiscriminationAgainstMuonHighEfficiency")
)

PFTausBothProngs = cms.EDAnalyzer("TauTagValidation",
   DataType                     = cms.string('Leptons'),
   MatchDeltaR_Leptons          = cms.double(0.15),
   MatchDeltaR_Jets             = cms.double(0.3),
   SaveOutputHistograms         = cms.bool(False),
   RefCollection                = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
   ExtensionName                = cms.string("ALLTHREE"),
   TauProducer                  = cms.string('pfRecoTauProducer'),
   TauProducerDiscriminators    = cms.untracked.vstring("pfRecoTauDiscriminationByLeadingTrackFinding","pfRecoTauDiscriminationByLeadingTrackPtCut","pfRecoTauDiscriminationByIsolation","pfRecoTauDiscriminationAgainstElectron","pfRecoTauDiscriminationAgainstMuon")
)

CaloTausBothProngs = cms.EDAnalyzer("TauTagValidation",
   DataType                     = cms.string('Leptons'),
   MatchDeltaR_Leptons          = cms.double(0.15),
   MatchDeltaR_Jets             = cms.double(0.3),
   SaveOutputHistograms         = cms.bool(False),
   RefCollection                = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
   ExtensionName                = cms.string("ALLTHREE"),
   TauProducer                  = cms.string('caloRecoTauProducer'),
   TauProducerDiscriminators    = cms.untracked.vstring("caloRecoTauDiscriminationByLeadingTrackFinding","caloRecoTauDiscriminationByLeadingTrackPtCut","caloRecoTauDiscriminationByIsolation","caloRecoTauDiscriminationAgainstElectron")
)

