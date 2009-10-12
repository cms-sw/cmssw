# This test config file is used to validate the fixedConePFTauProducer
# 

import FWCore.ParameterSet.Config as cms
import copy

StandardSignalParameters = cms.PSet(
   DataType                     = cms.string('Leptons'),               
   MatchDeltaR_Leptons          = cms.double(0.15),
   MatchDeltaR_Jets             = cms.double(0.3),
   SaveOutputHistograms         = cms.bool(False),
   RefCollection                = cms.InputTag("TauGenJetProducer","selectedGenTauDecaysToHadronsPt5Cumulative"),
)

def ChangeAModuleToBackground(aModule):
   ''' Used to switch a standard tau validation module from a signal configuration to a background configuration '''
   aModule.DataType = cms.string('Jets')
   aModule.RefCollection = cms.InputTag("IterativeConeJetProducer","iterativeCone5GenJets")

PFTausHighEfficiencyLeadingPionBothProngs = cms.EDAnalyzer("TauTagValidation",
   StandardSignalParameters,
   ExtensionName                = cms.string("LeadingPion"),
   TauProducer                  = cms.string('shrinkingConePFTauProducer'),
   discriminators               = cms.VPSet(
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByLeadingPionPtCut"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstMuon"),selectionCut = cms.double(0.5))
 )
)

PFTausHighEfficiencyBothProngs = cms.EDAnalyzer("TauTagValidation",
   StandardSignalParameters,
   ExtensionName                = cms.string(""),
   TauProducer                  = cms.string('shrinkingConePFTauProducer'),
   discriminators               = cms.VPSet(
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByLeadingTrackFinding"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByLeadingTrackPtCut"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTrackIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByECALIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstMuon"),selectionCut = cms.double(0.5))
 )
)

RunTancValidation = copy.deepcopy(PFTausHighEfficiencyBothProngs)
RunTancValidation.ExtensionName = "Tanc"
RunTancValidation.discriminators = cms.VPSet(
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByLeadingPionPtCut"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTaNCfrOnePercent"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTaNCfrHalfPercent"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationByTaNCfrTenthPercent"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("shrinkingConePFTauDiscriminationAgainstMuon"),selectionCut = cms.double(0.5))
)


PFTausBothProngs = cms.EDAnalyzer("TauTagValidation",
   StandardSignalParameters,
   ExtensionName                = cms.string(""),
   TauProducer                  = cms.string('fixedConePFTauProducer'),
   discriminators = cms.VPSet(
    cms.PSet( discriminator = cms.string("fixedConePFTauDiscriminationByLeadingTrackFinding"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("fixedConePFTauDiscriminationByLeadingTrackPtCut"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("fixedConePFTauDiscriminationByTrackIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("fixedConePFTauDiscriminationByECALIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("fixedConePFTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("fixedConePFTauDiscriminationAgainstMuon"),selectionCut = cms.double(0.5))
 )
)

CaloTausBothProngs = cms.EDAnalyzer("TauTagValidation",
   StandardSignalParameters,
   ExtensionName                = cms.string(""),
   TauProducer                  = cms.string('caloRecoTauProducer'),
   discriminators = cms.VPSet(
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationByLeadingTrackFinding"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationByLeadingTrackPtCut"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationByIsolation"),selectionCut = cms.double(0.5)),
    cms.PSet( discriminator = cms.string("caloRecoTauDiscriminationAgainstElectron"),selectionCut = cms.double(0.5))
 )
)

# Used to easily convert to background style.  
#  not necessary if you can iterate over a cms.Sequence (is it possible?)
TauTagValidationPackages = [
	PFTausBothProngs,
        CaloTausBothProngs,
        RunTancValidation,
        PFTausHighEfficiencyBothProngs,
        PFTausHighEfficiencyLeadingPionBothProngs,
        ]

tauTagValidation = cms.Sequence( 
	PFTausBothProngs+
        CaloTausBothProngs +
        PFTausHighEfficiencyBothProngs+
        PFTausHighEfficiencyLeadingPionBothProngs
	)

tauTagValidationWithTanc = cms.Sequence(
      tauTagValidation +
      RunTancValidation
      )
      
