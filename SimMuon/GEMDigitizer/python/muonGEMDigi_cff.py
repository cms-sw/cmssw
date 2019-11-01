import FWCore.ParameterSet.Config as cms

from SimMuon.GEMDigitizer.muonGEMDigis_cfi import *

muonGEMDigiTask = cms.Task(simMuonGEMDigis)
muonGEMDigi = cms.Sequence(muonGEMDigiTask)
