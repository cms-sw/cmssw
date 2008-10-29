# This test config file is used to validate the pfRecoTauProducer
# 

import FWCore.ParameterSet.Config as cms

generatorLevelJets = cms.EDAnalyzer("TauTagValidation",
   DataType                     = cms.string('Jets'),               
   MatchDeltaR_Leptons          = cms.double(0.15),
   MatchDeltaR_Jets             = cms.double(0.3),
   SaveOutputHistograms         = cms.bool(True),
   RefCollection                = cms.InputTag("GenJetProducer","GenJets"),
   ExtensionName                = cms.string(""),
   TauProducer                  = cms.string('pfRecoTauProducerHighEfficiency'),
   discriminators               = cms.VPSet(
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationByLeadingTrackFindingHighEfficiency"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationByLeadingTrackPtCutHighEfficiency"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationByIsolationHighEfficiency"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationAgainstElectronHighEfficiency"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("pfRecoTauDiscriminationAgainstMuonHighEfficiency"),selectionCut = cms.double(0.4))
 )

)

