import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
exoticaSingleJetHLT = hlt.hltHighLevel.clone()
exoticaSingleJetHLT.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
exoticaSingleJetHLT.HLTPaths = ['HLT_CentralJet80_MET65*','HLT_CentralJet80_MET80HF*']
exoticaSingleJetHLT.andOr = cms.bool( True )
exoticaSingleJetHLT.throw = cms.bool( False )

exoSingleJetSequence = cms.Sequence(exoticaSingleJetHLT)
