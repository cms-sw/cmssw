import FWCore.ParameterSet.Config as cms
from SimMuon.CSCDigitizer.muonCSCDigis_cfi import simMuonCSCDigis

simMuonCSCSuppressedDigis = cms.EDProducer("CSCDigiSuppressor",
    simMuonCSCDigis.strips,
    stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    lctTag = cms.InputTag("simCscTriggerPrimitiveDigis")
)

