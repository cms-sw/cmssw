import FWCore.ParameterSet.Config as cms
from SimMuon.CSCDigitizer.muonCSCDigis_cfi import simMuonCSCDigis

simMuonCSCSuppressedDigis = cms.EDProducer("CSCDigiSuppressor",
    simMuonCSCDigis.strips,
    digiLabel = cms.string("simMuonCSCDigis"),
    lctTag = cms.InputTag("simCscTriggerPrimitiveDigis")
)

