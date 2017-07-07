import FWCore.ParameterSet.Config as cms
import sys

from SimMuon.GEMDigitizer.me0ChamberMasker_cfi import me0ChamberMasker as _me0ChamberMasker
from SimMuon.GEMDigitizer.muonME0ReDigis_cfi import simMuonME0ReDigis

def appendME0ChamberMaskerAtReco(process):

    if hasattr(process,'RawToDigi') :

        sys.stderr.write("[appendME0ChamberMasker] : Found RawToDigi, appending filter\n")

        process.simMuonME0Digis = _me0ChamberMasker.clone()
        process.simMuonME0ReDigis = simMuonME0ReDigis.clone()
        process.simMuonME0Digis.digiTag =  cms.InputTag("simMuonME0Digis", \
                                                        processName = cms.InputTag.skipCurrentProcess())

        process.filteredME0DigiSequence = cms.Sequence( process.simMuonME0Digis 
                                                        + process.simMuonME0ReDigis )

        process.RawToDigi += process.filteredME0DigiSequence

    return process


