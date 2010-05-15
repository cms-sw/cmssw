import FWCore.ParameterSet.Config as cms
import copy
from Validation.RecoTau.RecoTauValidation_cfi import *

from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets
from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from Validation.RecoTau.ValidationOptions_cfi import *

# require generated tau to decay hadronically
objectTypeSelectedTauValDenominator = cms.EDFilter("TauGenJetDecayModeSelector",
     src = cms.InputTag("tauGenJets"),
     select = cms.vstring('oneProng0Pi0', 'oneProng1Pi0', 'oneProng2Pi0', 'oneProngOther',
                          'threeProng0Pi0', 'threeProng1Pi0', 'threeProngOther', 'rare'),
     filter = cms.bool(False)
)

#Here is where we call the value maps we want to use for efficiency
#WeightProducerTypes are : "shrinkingCone"
#WeightValueMapTypes are : "DiJetHighPt","DiJetSecondPt","MuEnrichedQCD","ZTT","WJets"
#WeightDiscriTypes   are : "ByCharge","ByNumTracks","ByLeadPionPt","ByChargeAndTracks","ByIsolation","AgainstElectron","AgainstMuon",
#                          "ByStandardChain","ByStandardChainNoElectron","ByStandardChainNoMuon"
#                          "ByTaNCChainTenth","ByTaNCChainQuarter","ByTaNCChainHalf","ByTaNCChainOne"
#                          "ByTaNCfrTenth","ByTaNCfrQuarter","ByTaNCfrHalf","ByTaNCfrOne",
#                          "ByTaNCChainTenthNoElectron","ByTaNCChainQuarterNoElectron","ByTaNCChainHalfNoElectron","ByTaNCChainOneNoElectron",
#                          "ByTaNCChainTenthNoMuon","ByTaNCChainQuarterNoMuon","ByTaNCChainHalfNoMuon","ByTaNCChainOneNoMuon",

ProducerTypeList = cms.vstring("shrinkingCone")

ValueMapTypeList = cms.vstring("ZTT")

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



PFTausHighEfficiencyBothProngsWithEfficiencies = copy.deepcopy(PFTausHighEfficiencyBothProngs)
PFTausHighEfficiencyBothProngsWithEfficiencies.WeightProducerType = cms.PSet(PType1 = cms.string("shrinkingCone")
                                                                          )
PFTausHighEfficiencyBothProngsWithEfficiencies.WeightValueMapType = cms.PSet(VMType1 = cms.string("ZTT"),
                                                                          )
PFTausHighEfficiencyBothProngsWithEfficiencies.WeightDiscriType   = cms.PSet(DType1 = cms.string("ByCharge"),
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
PFTausHighEfficiencyBothProngsWithEfficiencies.EventType = cms.string("%s" % options.eventType)




TauValNumeratorAndDenominatorWithEfficiencies = cms.Sequence(
      PFTausBothProngsWithEfficiencies +
      CaloTausBothProngsWithEfficiencies +
      PFTausHighEfficiencyBothProngsWithEfficiencies +
      PFTausHighEfficiencyLeadingPionBothProngsWithEfficiencies +
      RunTancValidationWithEfficiencies
      )

produceDenominator = cms.Sequence(
      tauGenJets
      +objectTypeSelectedTauValDenominator
      +kinematicSelectedTauValDenominator
      )

runTauValidationBatchMode = cms.Sequence(
      produceDenominator
      +TauValNumeratorAndDenominatorWithEfficiencies
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficiencies
      )

