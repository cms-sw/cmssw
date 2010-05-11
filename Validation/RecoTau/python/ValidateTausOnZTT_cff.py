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

PFTausBothProngsWithEfficiencies = copy.deepcopy(PFTausBothProngs)
PFTausBothProngsWithEfficiencies.WeightProducerType = cms.PSet()
PFTausBothProngsWithEfficiencies.WeightValueMapType = cms.PSet()
PFTausBothProngsWithEfficiencies.WeightDiscriType   = cms.PSet()
PFTausBothProngsWithEfficiencies.EventType = cms.string("%s" % options.eventType)

CaloTausBothProngsWithEfficiencies = copy.deepcopy(CaloTausBothProngs)
CaloTausBothProngsWithEfficiencies.WeightValueMapSource = cms.PSet()
CaloTausBothProngsWithEfficiencies.WeightProducerType = cms.PSet()
CaloTausBothProngsWithEfficiencies.WeightValueMapType = cms.PSet()
CaloTausBothProngsWithEfficiencies.WeightDiscriType   = cms.PSet()
CaloTausBothProngsWithEfficiencies.EventType = cms.string("%s" % options.eventType)


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


PFTausHighEfficiencyLeadingPionBothProngsWithEfficiencies = copy.deepcopy(PFTausHighEfficiencyLeadingPionBothProngs)
PFTausHighEfficiencyLeadingPionBothProngsWithEfficiencies.WeightProducerType = cms.PSet(PType1 = cms.string("shrinkingCone")
                                                                                     )
PFTausHighEfficiencyLeadingPionBothProngsWithEfficiencies.WeightValueMapType = cms.PSet(VMType1 = cms.string("ZTT")
                                                                                     )
PFTausHighEfficiencyLeadingPionBothProngsWithEfficiencies.WeightDiscriType   = cms.PSet(DType1 = cms.string("ByCharge"),
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
PFTausHighEfficiencyLeadingPionBothProngsWithEfficiencies.EventType = cms.string("%s" % options.eventType)


RunTancValidationWithEfficiencies = copy.deepcopy(RunTancValidation)
RunTancValidationWithEfficiencies.WeightProducerType = cms.PSet()
RunTancValidationWithEfficiencies.WeightValueMapType = cms.PSet()
RunTancValidationWithEfficiencies.WeightDiscriType   = cms.PSet()
RunTancValidationWithEfficiencies.EventType = cms.string("%s" % options.eventType)


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

