import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt

exoticaLongLivedMuHLT = hlt.hltHighLevel.clone()
exoticaLongLivedMuHLT.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
exoticaLongLivedMuHLT.HLTPaths = ['HLT_L2DoubleMu*_NoVertex_v*']
exoticaLongLivedMuHLT.andOr = cms.bool( True )
exoticaLongLivedMuHLT.throw = cms.bool( False )

exoticaLongLivedPhotonHLT = hlt.hltHighLevel.clone()
exoticaLongLivedPhotonHLT.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
exoticaLongLivedPhotonHLT.HLTPaths = ['HLT_DoublePhoton33_v*']
exoticaLongLivedPhotonHLT.andOr = cms.bool( True )
exoticaLongLivedPhotonHLT.throw = cms.bool( False )

exoticaLongLivedHLT = hlt.hltHighLevel.clone()
exoticaLongLivedHLT.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
exoticaLongLivedHLT.HLTPaths = ['HLT_L2DoubleMu*_NoVertex_v*', 'HLT_DoublePhoton33_v*']
exoticaLongLivedHLT.andOr = cms.bool( True )
exoticaLongLivedHLT.throw = cms.bool( False )

exoLongLivedMuSequence = cms.Sequence(exoticaLongLivedMuHLT)
exoLongLivedPhotonSequence = cms.Sequence(exoticaLongLivedPhotonHLT)
#exoLongLivedSequence = cms.Sequence(exoticaLongLivedMuHLT * exoticaLongLivedPhotonHLT)
exoLongLivedSequence = cms.Sequence(exoticaLongLivedHLT)
