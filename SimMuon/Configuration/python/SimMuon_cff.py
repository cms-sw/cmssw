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
from SimMuon.RPCDigitizer.muonrpcdigi_cfi import *
muonDigi = cms.Sequence(simMuonCSCDigis+simMuonDTDigis+simMuonRPCDigis)

from CondCore.DBCommon.CondDBCommon_cfi import *

from SimMuon.GEMDigitizer.muonGEMDigi_cff import *
from SimMuon.GEMDigitizer.muonME0Digi_cff import *

_run3_muonDigi = muonDigi.copy()
_run3_muonDigi += muonGEMDigi

_phase2_muonDigi = _run3_muonDigi.copy()
_phase2_muonDigi += muonME0Digi

def _modifySimMuonForPhase2( theProcess ):
    theProcess.rpcphase2recovery_essource = cms.ESSource("PoolDBESSource",
         CondDBCommon,
         #    using CondDBSetup
         toGet = cms.VPSet(cms.PSet(
             record = cms.string("RPCStripNoisesRcd"),             
             tag = cms.string("RPC_testCondition_192Strips_mc"),
             connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
            ),
         cms.PSet(record = cms.string("RPCClusterSizeRcd"),
                 tag = cms.string("RPCClusterSize_PhaseII_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
                 )
         )
    )
    theProcess.rpcphase2recovery_esprefer = cms.ESPrefer("PoolDBESSource","rpcphase2recovery_essource")

from from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toReplaceWith( muonDigi, _run3_muonDigi )
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith( muonDigi, _run3_muonDigi )
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith( muonDigi, _phase2_muonDigi )
modifyConfigurationStandardSequencesSimMuonPhase2_ = phase2_muon.makeProcessModifier( _modifySimMuonForPhase2 )
