import FWCore.ParameterSet.Config as cms

# Module to create simulated GEM-CSC trigger pad digis.
simMuonGEMPadDigis = cms.EDProducer("GEMPadDigiProducer",
    InputCollection = cms.InputTag('simMuonGEMDigis'),
)
