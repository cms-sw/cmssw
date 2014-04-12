import FWCore.ParameterSet.Config as cms

muonSelectorVertex = cms.EDProducer(
  "MuonSelectorVertex"
, muonSource   = cms.InputTag( 'selectedPatMuons' )
, vertexSource = cms.InputTag( 'offlinePrimaryVertices' )
, maxDZ = cms.double( 0.5 )
)
