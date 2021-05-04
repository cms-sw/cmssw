import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0

from SimMuon.GEMDigitizer.muonME0Digis_cfi import *
from SimMuon.GEMDigitizer.muonME0PseudoDigis_cfi import *
from SimMuon.GEMDigitizer.muonME0PseudoReDigis_cfi import *

muonME0RealDigi = cms.Task(simMuonME0Digis)

muonME0PseudoDigi = cms.Task(simMuonME0PseudoDigis, simMuonME0PseudoReDigis)

muonME0DigiTask = cms.Task(muonME0RealDigi, muonME0PseudoDigi)

## in scenarios with GE0, remove the pseudo digis
phase2_GE0.toReplaceWith(muonME0DigiTask, muonME0DigiTask.copyAndExclude([muonME0PseudoDigi]))

muonME0Digi = cms.Sequence(muonME0DigiTask)
