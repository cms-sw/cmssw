# This test config file is used to validate the pfRecoTauProducer
# 

import FWCore.ParameterSet.Config as cms

PFTausHighEfficiencyBothProngs = cms.EDAnalyzer("TauTagValidation",
   DataType                     = cms.string('Leptons'),               
   MatchDeltaR_Leptons          = cms.double(0.15),
   MatchDeltaR_Jets             = cms.double(0.3),
   SaveOutputHistograms         = cms.bool(False),
   RefCollection                = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
   ExtensionName                = cms.string(""),
   TauProducer                  = cms.string('pfRecoTauProducerHighEfficiency'),
   discriminators               = cms.VPSet(
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationByLeadingTrackFindingHighEfficiency"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationByLeadingTrackPtCutHighEfficiency"),selectionCut = cms.double(0.5)),
#    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationByIsolationHighEfficiency"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationByTrackIsolationHighEfficiency"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationByECALIsolationHighEfficiency"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationAgainstElectronHighEfficiency"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationAgainstMuonHighEfficiency"),selectionCut = cms.double(0.4))
 )
)

PFTausBothProngs = cms.EDAnalyzer("TauTagValidation",
   DataType                     = cms.string('Leptons'),
   MatchDeltaR_Leptons          = cms.double(0.15),
   MatchDeltaR_Jets             = cms.double(0.3),
   SaveOutputHistograms         = cms.bool(False),
   RefCollection                = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
   ExtensionName                = cms.string(""),
   TauProducer                  = cms.string('pfRecoTauProducer'),
   discriminators = cms.VPSet(
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationByLeadingTrackFinding"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationByLeadingTrackPtCut"),selectionCut = cms.double(0.5)),
#    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationByIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationByTrackIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationByECALIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationAgainstMuon"),selectionCut = cms.double(0.4))
 )
)

CaloTausBothProngs = cms.EDAnalyzer("TauTagValidation",
   DataType                     = cms.string('Leptons'),
   MatchDeltaR_Leptons          = cms.double(0.15),
   MatchDeltaR_Jets             = cms.double(0.3),
   SaveOutputHistograms         = cms.bool(True),
   RefCollection                = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
   ExtensionName                = cms.string("NewDiscriminators"),
   TauProducer                  = cms.string('caloRecoTauProducer'),
   discriminators = cms.VPSet(
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationByLeadingTrackFinding"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationByLeadingTrackPtCut"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationByIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5))
 )
)

