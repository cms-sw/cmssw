import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *
import copy

#from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from RecoJets.JetProducers.ic5PFJets_cfi import iterativeCone5PFJets
#from RecoJets.Configuration.RecoGenJets_cff import *
#from RecoJets.Configuration.GenJetParticles_cff import *

objectTypeSelectedTauValDenominator = copy.deepcopy(iterativeCone5PFJets)

PFTausBothProngsWithFakeRates = copy.deepcopy(PFTausBothProngs)
PFTausBothProngsWithFakeRates.WeightValueMapSource = cms.PSet()
PFTausBothProngsWithFakeRates.WeightProducerType = cms.PSet()
PFTausBothProngsWithFakeRates.WeightValueMapType = cms.PSet()
PFTausBothProngsWithFakeRates.WeightDiscriType   = cms.PSet()
PFTausBothProngsWithFakeRates.EventType = cms.string("%s" % options.eventType)

CaloTausBothProngsWithFakeRates = copy.deepcopy(CaloTausBothProngs)
CaloTausBothProngsWithFakeRates.WeightValueMapSource = cms.PSet()
CaloTausBothProngsWithFakeRates.WeightProducerType = cms.PSet()
CaloTausBothProngsWithFakeRates.WeightValueMapType = cms.PSet()
CaloTausBothProngsWithFakeRates.WeightDiscriType   = cms.PSet()
CaloTausBothProngsWithFakeRates.EventType = cms.string("%s" % options.eventType)


#Here is where we call the value maps we want to use for fake rate
#WeightProducerType are : "shrinkingCone"
#WeightValueMapType are : "DiJetHighPt","DiJetSecondPt","MuEnrichedQCD","ZTT","WJets"
#WeightDiscriType   are : "ByCharge","ByNumTracks","ByLeadPionPt","ByChargeAndTracks","ByIsolation","AgainstElectron","AgainstMuon",
#                         "ByStandardChain","ByStandardChainNoElectron","ByStandardChainNoMuon"
#                         "ByTaNCChainTenth","ByTaNCChainQuarter","ByTaNCChainHalf","ByTaNCChainOne"
#                         "ByTaNCfrTenth","ByTaNCfrQuarter","ByTaNCfrHalf","ByTaNCfrOne",
#                         "ByTaNCChainTenthNoElectron","ByTaNCChainQuarterNoElectron","ByTaNCChainHalfNoElectron","ByTaNCChainOneNoElectron",
#                         "ByTaNCChainTenthNoMuon","ByTaNCChainQuarterNoMuon","ByTaNCChainHalfNoMuon","ByTaNCChainOneNoMuon",
#The names (i.e. PType(n), VPType(n), DType(n)) can be anything you want, it does not matters ;)

PFTausHighEfficiencyBothProngsWithFakeRates = copy.deepcopy(PFTausHighEfficiencyBothProngs)
PFTausHighEfficiencyBothProngsWithFakeRates.WeightProducerType = cms.PSet(PType1 = cms.string("shrinkingCone")
                                                                          )
PFTausHighEfficiencyBothProngsWithFakeRates.WeightValueMapType = cms.PSet(VMType1 = cms.string("DiJetHighPt"),
                                                                          VMType2 = cms.string("DiJetSecondPt"),
                                                                          VMType3 = cms.string("MuEnrichedQCD")
                                                                          )
PFTausHighEfficiencyBothProngsWithFakeRates.WeightDiscriType   = cms.PSet(DType1 = cms.string("ByCharge"),
                                                                          DType2 = cms.string("ByNumTracks"),
                                                                          DType3 = cms.string("ByLeadPionPt"),
                                                                          DType4 = cms.string("ByLeadTrackPt"),
                                                                          DType5 = cms.string("ByChargeAndTracks"),
                                                                          DType6 = cms.string("ByIsolation"),
                                                                          DType7 = cms.string("AgainstElectron"),
                                                                          DType8 = cms.string("AgainstMuon"),
                                                                          DType9 = cms.string("ByStandardChain"),
                                                                          DType10 = cms.string("ByStandardChainNoElectron"),
                                                                          DType11 = cms.string("ByStandardChainNoMuon")
                                                                          )
PFTausHighEfficiencyBothProngsWithFakeRates.EventType = cms.string("%s" % options.eventType)


PFTausHighEfficiencyLeadingPionBothProngsWithFakeRates = copy.deepcopy(PFTausHighEfficiencyLeadingPionBothProngs)
PFTausHighEfficiencyLeadingPionBothProngsWithFakeRates.WeightProducerType = cms.PSet(PType1 = cms.string("shrinkingCone")
                                                                                     )
PFTausHighEfficiencyLeadingPionBothProngsWithFakeRates.WeightValueMapType = cms.PSet(VMType1 = cms.string("DiJetHighPt"),
                                                                                     VMType2 = cms.string("DiJetSecondPt"),
                                                                                     VMType3 = cms.string("MuEnrichedQCD")
                                                                                     )
PFTausHighEfficiencyLeadingPionBothProngsWithFakeRates.WeightDiscriType   = cms.PSet(DType1 = cms.string("ByCharge"),
                                                                                     DType2 = cms.string("ByNumTracks"),
                                                                                     DType3 = cms.string("ByLeadPionPt"),
                                                                                     DType4 = cms.string("ByLeadTrackPt"),
                                                                                     DType5 = cms.string("ByChargeAndTracks"),
                                                                                     DType6 = cms.string("ByIsolation"),
                                                                                     DType7 = cms.string("AgainstElectron"),
                                                                                     DType8 = cms.string("AgainstMuon"),
                                                                                     DType9 = cms.string("ByStandardChain"),
                                                                                     DType10 = cms.string("ByStandardChainNoElectron"),
                                                                                     DType11 = cms.string("ByStandardChainNoMuon"),
                                                                                     )
PFTausHighEfficiencyLeadingPionBothProngsWithFakeRates.EventType = cms.string("%s" % options.eventType)


RunTancValidationWithFakeRates = copy.deepcopy(RunTancValidation)
RunTancValidationWithFakeRates.WeightProducerType = cms.PSet()
RunTancValidationWithFakeRates.WeightValueMapType = cms.PSet()
RunTancValidationWithFakeRates.WeightDiscriType   = cms.PSet()
RunTancValidationWithFakeRates.EventType = cms.string("%s" % options.eventType)


TauValNumeratorAndDenominatorWithFakeRates = cms.Sequence(
      PFTausBothProngsWithFakeRates +
      CaloTausBothProngsWithFakeRates +
      PFTausHighEfficiencyBothProngsWithFakeRates +
      PFTausHighEfficiencyLeadingPionBothProngsWithFakeRates +
      RunTancValidationWithFakeRates
      )

produceDenominator = cms.Sequence(
      #iterativeCone5PFJets
      objectTypeSelectedTauValDenominator
      *kinematicSelectedTauValDenominatorForRealData
      )

runTauValidationBatchMode = cms.Sequence(
      produceDenominator
      +TauValNumeratorAndDenominatorWithFakeRates
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficiencies
      )
