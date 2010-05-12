import FWCore.ParameterSet.Config as cms
import copy

from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from RecoJets.Configuration.RecoGenJets_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *
from Validation.RecoTau.ValidationOptions_cfi import *

objectTypeSelectedTauValDenominator = copy.deepcopy(iterativeCone5GenJets)


#Here is where we call the value maps we want to use for fake rate
#WeightProducerTypes are : "shrinkingCone"
#WeightValueMapTypes are : "DiJetHighPt","DiJetSecondPt","MuEnrichedQCD","ZTT","WJets"
#WeightDiscriTypes   are : "ByCharge","ByNumTracks","ByLeadPionPt","ByChargeAndTracks","ByIsolation","AgainstElectron","AgainstMuon",
#                          "ByStandardChain","ByStandardChainNoElectron","ByStandardChainNoMuon"
#                          "ByTaNCChainTenth","ByTaNCChainQuarter","ByTaNCChainHalf","ByTaNCChainOne"
#                          "ByTaNCfrTenth","ByTaNCfrQuarter","ByTaNCfrHalf","ByTaNCfrOne",
#                          "ByTaNCChainTenthNoElectron","ByTaNCChainQuarterNoElectron","ByTaNCChainHalfNoElectron","ByTaNCChainOneNoElectron",
#                          "ByTaNCChainTenthNoMuon","ByTaNCChainQuarterNoMuon","ByTaNCChainHalfNoMuon","ByTaNCChainOneNoMuon",

ProducerTypeList = cms.vstring("shrinkingCone")

ValueMapTypeList = cms.vstring("DiJetHighPt",
                               "DiJetSecondPt",
                               "MuEnrichedQCD")

DiscriTypeList =  cms.vstring("ByCharge",
                              "ByNumTracks",
                              "ByLeadPionPt",
                              "ByLeadTrackPt",
                              "ByChargeAndTracks",
                              "ByIsolation",
                              "AgainstElectron",
                              "AgainstMuon",
                              "ByStandardChain",
                              "ByStandardChainNoElectron",
                              "ByStandardChainNoMuon")

EmptyList = cms.vstring()
                                                                          
from Validation.RecoTau.RecoTauValidation_cfi import *


PFTausBothProngsWithFakeRates = copy.deepcopy(PFTausBothProngs)
PFTausBothProngsWithFakeRates.WeightValueMapSource = EmptyList
PFTausBothProngsWithFakeRates.WeightProducerType = EmptyList
PFTausBothProngsWithFakeRates.WeightValueMapType = EmptyList
PFTausBothProngsWithFakeRates.WeightDiscriType   = EmptyList
PFTausBothProngsWithFakeRates.EventType = cms.string("%s" % options.eventType)

CaloTausBothProngsWithFakeRates = copy.deepcopy(CaloTausBothProngs)
CaloTausBothProngsWithFakeRates.WeightValueMapSource = EmptyList
CaloTausBothProngsWithFakeRates.WeightProducerType = EmptyList
CaloTausBothProngsWithFakeRates.WeightValueMapType = EmptyList
CaloTausBothProngsWithFakeRates.WeightDiscriType   = EmptyList
CaloTausBothProngsWithFakeRates.EventType = cms.string("%s" % options.eventType)


PFTausHighEfficiencyBothProngsWithFakeRates = copy.deepcopy(PFTausHighEfficiencyBothProngs)
PFTausHighEfficiencyBothProngsWithFakeRates.WeightProducerType = ProducerTypeList
PFTausHighEfficiencyBothProngsWithFakeRates.WeightValueMapType = ValueMapTypeList
PFTausHighEfficiencyBothProngsWithFakeRates.WeightDiscriType   = DiscriTypeList
PFTausHighEfficiencyBothProngsWithFakeRates.EventType = cms.string("%s" % options.eventType)


PFTausHighEfficiencyLeadingPionBothProngsWithFakeRates = copy.deepcopy(PFTausHighEfficiencyLeadingPionBothProngs)
PFTausHighEfficiencyLeadingPionBothProngsWithFakeRates.WeightProducerType = ProducerTypeList
PFTausHighEfficiencyLeadingPionBothProngsWithFakeRates.WeightValueMapType = ValueMapTypeList
PFTausHighEfficiencyLeadingPionBothProngsWithFakeRates.WeightDiscriType   = DiscriTypeList
PFTausHighEfficiencyLeadingPionBothProngsWithFakeRates.EventType = cms.string("%s" % options.eventType)


RunTancValidationWithFakeRates = copy.deepcopy(RunTancValidation)
RunTancValidationWithFakeRates.WeightProducerType = EmptyList
RunTancValidationWithFakeRates.WeightValueMapType = EmptyList
RunTancValidationWithFakeRates.WeightDiscriType   = EmptyList
RunTancValidationWithFakeRates.EventType = cms.string("%s" % options.eventType)



TauValNumeratorAndDenominatorWithFakeRates = cms.Sequence(
      PFTausBothProngsWithFakeRates +
      CaloTausBothProngsWithFakeRates +
      PFTausHighEfficiencyBothProngsWithFakeRates +
      PFTausHighEfficiencyLeadingPionBothProngsWithFakeRates +
      RunTancValidationWithFakeRates
      )

produceDenominator = cms.Sequence(
      genParticlesForJets
      *objectTypeSelectedTauValDenominator
      *kinematicSelectedTauValDenominator
      )

runTauValidationBatchMode = cms.Sequence(
      produceDenominator
      +TauValNumeratorAndDenominatorWithFakeRates
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficiencies
      )
