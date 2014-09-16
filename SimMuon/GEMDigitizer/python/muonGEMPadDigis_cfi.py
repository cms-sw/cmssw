import FWCore.ParameterSet.Config as cms

# Module to create simulated GEM-CSC trigger pad digis.
simMuonGEMCSCPadDigis = cms.EDProducer("GEMCSCPadDigiProducer",
    InputCollection = cms.InputTag('simMuonGEMDigis'),
)
