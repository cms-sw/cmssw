import FWCore.ParameterSet.Config as cms

matchedGsfElectrons = cms.EDProducer('GenMatchedElectronsProducer',
  src = cms.InputTag('gsfElectrons'),
  match = cms.InputTag('electronMatch')
)
