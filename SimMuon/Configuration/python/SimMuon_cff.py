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
from SimMuon.GEMDigitizer.muonME0DigisPreReco_cfi import *

_phase2_muonDigi = muonDigi.copy()
_phase2_muonDigi += muonGEMDigi
_phase2_muonDigi += simMuonME0Digis

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

from Configuration.StandardSequences.Eras import eras
eras.phase2_muon.toReplaceWith( muonDigi, _phase2_muonDigi )
modifyConfigurationStandardSequencesSimMuonPhase2_ = eras.phase2_muon.makeProcessModifier( _modifySimMuonForPhase2 )
