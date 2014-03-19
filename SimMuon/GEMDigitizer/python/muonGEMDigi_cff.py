import FWCore.ParameterSet.Config as cms

from SimMuon.GEMDigitizer.muonGEMDigis_cfi import *
from SimMuon.GEMDigitizer.muonGEMCSCPadDigis_cfi import *

muonGEMDigi = cms.Sequence(simMuonGEMDigis*simMuonGEMCSCPadDigis)
