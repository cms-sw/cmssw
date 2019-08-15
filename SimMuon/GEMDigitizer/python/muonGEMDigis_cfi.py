import FWCore.ParameterSet.Config as cms

from SimMuon.GEMDigitizer.simMuonGEMDigisDef_cfi import *
simMuonGEMDigis = simMuonGEMDigisDef.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(simMuonGEMDigis, mixLabel = "mixData")

from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( simMuonGEMDigis, instLumi = 1.5)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify( simMuonGEMDigis, instLumi = 2.0)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify( simMuonGEMDigis, instLumi = 5)
