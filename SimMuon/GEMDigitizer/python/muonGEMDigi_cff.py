import FWCore.ParameterSet.Config as cms

from SimMuon.GEMDigitizer.muonGEMDigis_cfi import *
from SimMuon.GEMDigitizer.muonGEMPadDigis_cfi import *
from SimMuon.GEMDigitizer.muonGEMPadDigiClusters_cfi import *

muonGEMDigi = cms.Sequence(simMuonGEMDigis*simMuonGEMPadDigis*simMuonGEMPadDigiClusters)
