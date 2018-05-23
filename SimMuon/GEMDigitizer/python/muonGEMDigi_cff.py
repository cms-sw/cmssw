import FWCore.ParameterSet.Config as cms

from SimMuon.GEMDigitizer.muonGEMDigis_cfi import *
from SimMuon.GEMDigitizer.muonGEMPadDigis_cfi import *
from SimMuon.GEMDigitizer.muonGEMPadDigiClusters_cfi import *

muonGEMDigi = cms.Sequence(simMuonGEMDigis*simMuonGEMPadDigis*simMuonGEMPadDigiClusters)

# In phase2 premixing we use simMuonGEMDigis for the overlay, and the rest are run after that
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
(premix_stage2 & phase2_muon).toReplaceWith(muonGEMDigi, cms.Sequence(simMuonGEMDigis))

muonGEMDigiDM = cms.Sequence(simMuonGEMPadDigis*simMuonGEMPadDigiClusters)
