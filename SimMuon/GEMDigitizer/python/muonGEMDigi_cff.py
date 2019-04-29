import FWCore.ParameterSet.Config as cms

from SimMuon.GEMDigitizer.muonGEMDigis_cfi import *
from SimMuon.GEMDigitizer.muonGEMPadDigis_cfi import *
from SimMuon.GEMDigitizer.muonGEMPadDigiClusters_cfi import *

muonGEMDigiTask = cms.Task(simMuonGEMDigis, simMuonGEMPadDigis, simMuonGEMPadDigiClusters)
muonGEMDigi = cms.Sequence(muonGEMDigiTask)
