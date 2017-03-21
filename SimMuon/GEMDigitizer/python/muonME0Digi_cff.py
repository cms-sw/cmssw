import FWCore.ParameterSet.Config as cms

from SimMuon.GEMDigitizer.muonME0DigisPreReco_cfi import *
from SimMuon.GEMDigitizer.muonME0ReDigis_cfi import *

muonME0Digi = cms.Sequence(simMuonME0Digis*simMuonME0ReDigis)
