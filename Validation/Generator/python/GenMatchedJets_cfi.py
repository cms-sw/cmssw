import FWCore.ParameterSet.Config as cms

partonMatchedJets = cms.EDProducer('PartonMatchedJetsProducer',
  src = cms.InputTag('antikt5CaloJets'),
  match = cms.InputTag('jetPartonMatch')
)

genJetMatchedJets = cms.EDProducer('GenJetMatchedJetsProducer',
  src = cms.InputTag('antikt5CaloJets'),
  match = cms.InputTag('jetGenJetMatch')
)
