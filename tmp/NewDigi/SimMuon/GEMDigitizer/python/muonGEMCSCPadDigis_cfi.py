import FWCore.ParameterSet.Config as cms

# Module to create simulated GEM-CSC trigger pad digis.
simMuonGEMCSCPadDigis = cms.EDProducer("GEMCSCPadDigiProducer",
    inputCollection = cms.InputTag('simMuonGEMDigis'),
    maxDeltaBX = cms.int32(1)
)
