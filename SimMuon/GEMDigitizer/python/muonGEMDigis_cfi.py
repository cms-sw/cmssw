import FWCore.ParameterSet.Config as cms

gemDigiCommonParameters = cms.PSet(
    instLumi = cms.double(7.5), # in units of 1E34 cm^-2 s^-1. Internally the background is parametrized from FLUKA+GEANT results at 5x10^34 (PU140). We are adding a 1.5 factor for PU200
    rateFact = cms.double(1.0), # Set this factor to 1 since the new background model includes the new beam pipe and the relevant effects, so no need of higher safety factor. keeping is here is just for backward compatibiliy
    referenceInstLumi = cms.double(5.), #In units of 10^34 Hz/cm^2. Internally the functions based on the FLUKA+GEANT simulation are normalized to 5x10^34 Hz/cm^2, this is needed to rescale them properly
)

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
