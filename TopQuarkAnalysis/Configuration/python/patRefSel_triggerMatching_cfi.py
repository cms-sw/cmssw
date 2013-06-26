import FWCore.ParameterSet.Config as cms

patMuonTriggerMatch = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"
, src     = cms.InputTag( "cleanPatMuons" )
, matched = cms.InputTag( "patTrigger" )
, matchedCuts = cms.string( '' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )
, resolveByMatchQuality = cms.bool( True )
)

patPhotonTriggerMatch = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"
, src     = cms.InputTag( "cleanPatPhotons" )
, matched = cms.InputTag( "patTrigger" )
, matchedCuts = cms.string( '' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )
, resolveByMatchQuality = cms.bool( True )
)

patElectronTriggerMatch = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"
, src     = cms.InputTag( "cleanPatElectrons" )
, matched = cms.InputTag( "patTrigger" )
, matchedCuts = cms.string( '' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )
, resolveByMatchQuality = cms.bool( True )
)

patTauTriggerMatch = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"
, src     = cms.InputTag( "cleanPatTaus" )
, matched = cms.InputTag( "patTrigger" )
, matchedCuts = cms.string( '' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )
, resolveByMatchQuality = cms.bool( True )
)

patJetTriggerMatch = cms.EDProducer(
  "PATTriggerMatcherDRLessByR"
, src     = cms.InputTag( "cleanPatJets" )
, matched = cms.InputTag( "patTrigger" )
, matchedCuts = cms.string( '' )
, maxDeltaR = cms.double( 0.4 )
, resolveAmbiguities    = cms.bool( True )
, resolveByMatchQuality = cms.bool( True )
)

patMETTriggerMatch = cms.EDProducer(
  "PATTriggerMatcherDRLessByR"
, src     = cms.InputTag( "patMETs" )
, matched = cms.InputTag( "patTrigger" )
, matchedCuts = cms.string( '' )
, maxDeltaR = cms.double( 0.4 )
, resolveAmbiguities    = cms.bool( True )
, resolveByMatchQuality = cms.bool( True )
)
