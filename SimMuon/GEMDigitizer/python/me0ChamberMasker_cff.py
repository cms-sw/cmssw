import FWCore.ParameterSet.Config as cms
import sys

from SimMuon.GEMDigitizer.me0ChamberMasker_cfi import me0ChamberMasker as _me0ChamberMasker
from SimMuon.GEMDigitizer.muonME0PseudoReDigis_cfi import simMuonME0PseudoReDigis

def appendME0ChamberMaskerAtReco(process):

    if hasattr(process,'RawToDigi') :

        sys.stderr.write("[appendME0ChamberMasker] : Found RawToDigi, appending filter\n")

        process.simMuonME0PseudoDigis = _me0ChamberMasker.clone()
        process.simMuonME0PseudoReDigis = simMuonME0PseudoReDigis.clone()
        process.simMuonME0PseudoDigis.digiTag =  cms.InputTag("simMuonME0PseudoDigis", \
                                                        processName = cms.InputTag.skipCurrentProcess())

        process.filteredME0DigiSequence = cms.Sequence( process.simMuonME0PseudoDigis 
                                                        + process.simMuonME0PseudoReDigis )

        process.RawToDigi += process.filteredME0DigiSequence

    return process
