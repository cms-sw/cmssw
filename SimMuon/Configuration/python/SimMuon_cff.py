import FWCore.ParameterSet.Config as cms

# Muon Digitization (CSC, DT, RPC electronics responce)

# CSC digitizer
#
from SimMuon.CSCDigitizer.muonCSCDigis_cfi import *
from CalibMuon.CSCCalibration.CSCChannelMapper_cfi import *
from CalibMuon.CSCCalibration.CSCIndexer_cfi import *

# DT digitizer
#
from SimMuon.DTDigitizer.muondtdigi_cfi import *

# RPC digitizer
#
# If your branch still uses the legacy filename muonrpcdigi_cfi.py,
# change only this import line back to that filename.
from SimMuon.RPCDigitizer.muonRPCDigis_cfi import *

muonDigiTask = cms.Task(simMuonCSCDigis, simMuonDTDigis, simMuonRPCDigis)
muonDigi = cms.Sequence(muonDigiTask)

from SimMuon.GEMDigitizer.muonGEMDigi_cff import *
from SimMuon.GEMDigitizer.muonME0Digi_cff import *

_run3_muonDigiTask = muonDigiTask.copy()
_run3_muonDigiTask.add(muonGEMDigiTask)

_phase2_muonDigiTask = _run3_muonDigiTask.copy()
_phase2_muonDigiTask.add(muonME0DigiTask)

# while GE0 is in development, just turn off ME0 tasks
_phase2_ge0 = _phase2_muonDigiTask.copyAndExclude([muonME0DigiTask])

# phase2_rpc_devel:
# keep legacy simMuonRPCDigis for side-by-side validation
# and add the new Phase-2 RPC + iRPC digis
_phase2_rpc_devel = _phase2_muonDigiTask.copyAndExclude([muonME0DigiTask])
_phase2_rpc_devel.add(simMuonRPCDigisPhase2)
_phase2_rpc_devel.add(simMuonIRPCDigis)


from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toReplaceWith( muonDigiTask, _run3_muonDigiTask )
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith( muonDigiTask, _run3_muonDigiTask )
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith( muonDigiTask, _phase2_muonDigiTask )
from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0
phase2_GE0.toReplaceWith( muonDigiTask, _phase2_ge0 )
from Configuration.Eras.Modifier_phase2_rpc_devel_cff import phase2_rpc_devel
phase2_rpc_devel.toReplaceWith(muonDigiTask, _phase2_rpc_devel)