import FWCore.ParameterSet.Config as cms

# Module to create simulated GEM-CSC trigger pad digis.
simMuonGEMPadDigis = cms.EDProducer("GEMPadDigiProducer",
    InputCollection = cms.InputTag('simMuonGEMDigis'),
)

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
(premix_stage2 & phase2_muon).toModify(simMuonGEMPadDigis, InputCollection = "mixData")
