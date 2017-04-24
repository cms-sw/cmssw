import FWCore.ParameterSet.Config as cms

from SimMuon.DTDigitizer.dtChamberMasker_cff  import *
from SimMuon.RPCDigitizer.rpcChamberMasker_cff import *
from SimMuon.CSCDigitizer.cscChamberMasker_cff import *
from SimMuon.GEMDigitizer.gemChamberMasker_cff import *
from SimMuon.GEMDigitizer.me0ChamberMasker_cff import *

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon

def enableAgingAtReco(process):
    "Enable muon masking/aging for digis entering muon reconstruction"    
  

    appendCSCChamberMaskerAtUnpacking(process)
    appendDTChamberMaskerAtUnpacking(process) 
    if phase2_muon.isChosen():
        appendRPCChamberMaskerBeforeRecHits(process)
        appendGEMChamberMaskerAtReco(process)    
        appendME0ChamberMaskerAtReco(process)    
    else :
        appendRPCChamberMaskerAtUnpacking(process)

    return process
