import FWCore.ParameterSet.Config as cms

muonTriggerMatch = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"
, src     = cms.InputTag( 'signalMuons' )
, matched = cms.InputTag( 'patTrigger' )
, matchedCuts = cms.string( 'type("TriggerMuon")' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )
, resolveByMatchQuality = cms.bool( True )
)

signalMuonsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchMuonEmbedder"
, src = cms.InputTag( 'signalMuons' )
, matches = cms.VInputTag(
    cms.InputTag( 'muonTriggerMatch' )
  )
)
