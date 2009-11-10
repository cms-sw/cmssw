import FWCore.ParameterSet.Config as cms

matchedMuons = cms.EDProducer('GenMatchedMuonsProducer',
  src = cms.InputTag('muons'),
  match = cms.InputTag('muonMatch')
)
