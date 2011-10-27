import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt

exoticaHT450 = hlt.hltHighLevel.clone()
exoticaHT450.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
exoticaHT450.HLTPaths = ['HLT_HT450_v*']
exoticaHT450.andOr = cms.bool( True )
exoticaHT450.throw = cms.bool( False )

exoticaHT500 = hlt.hltHighLevel.clone()
exoticaHT500.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
exoticaHT500.HLTPaths = ['HLT_HT500_v*']
exoticaHT500.andOr = cms.bool( True )
exoticaHT500.throw = cms.bool( False )

exoticaHT = hlt.hltHighLevel.clone()
exoticaHT.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
exoticaHT.HLTPaths = ['HLT_HT450_v*','HLT_HT500_v*']
exoticaHT.andOr = cms.bool( True )
exoticaHT.throw = cms.bool( False )

exoHTSequence = cms.Sequence(exoticaHT)
