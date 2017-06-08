import FWCore.ParameterSet.Config as cms

from SimMuon.DTDigitizer.dtChamberMasker_cff  import *
from SimMuon.RPCDigitizer.rpcChamberMasker_cff import *
from SimMuon.CSCDigitizer.cscChamberMasker_cff import *
from SimMuon.GEMDigitizer.gemChamberMasker_cff import *
from SimMuon.GEMDigitizer.me0ChamberMasker_cff import *

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon

# NOTES:
# - presently intended to work only in upgarde scenarios including GEM/ME0
# - applying aging before the RECO step breaks reproducibility (unless a failure == 0 eff is simulated)
#   NEEDS to be updated to apply aging after DIGIs are produced (to restore reproducibility of RECO 
#   and propagarte aged digis consistently everywhere, i.e. also in trigger)

def enableAgingAtReco(process):
    "Enable muon masking/aging for digis entering muon reconstruction"    

    appendCSCChamberMaskerAtUnpacking(process)
    appendDTChamberMaskerAtUnpacking(process) 
    appendRPCChamberMaskerBeforeRecHits(process)
    appendGEMChamberMaskerAtReco(process)    
    appendME0ChamberMaskerAtReco(process)    

    return process
