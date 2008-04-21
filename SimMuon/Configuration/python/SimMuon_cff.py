import FWCore.ParameterSet.Config as cms

from SimMuon.CSCDigitizer.muonCSCDigis_cfi import *
from SimMuon.DTDigitizer.muondtdigi_cfi import *
from SimMuon.RPCDigitizer.muonrpcdigi_cfi import *
muonDigi = cms.Sequence(muonCSCDigis+muonDTDigis+muonRPCDigis)

