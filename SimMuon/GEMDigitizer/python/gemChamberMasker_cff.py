import FWCore.ParameterSet.Config as cms
import sys

from SimMuon.GEMDigitizer.gemChamberMasker_cfi import gemChamberMasker as _gemChamberMasker
from SimMuon.GEMDigitizer.muonGEMPadDigis_cfi import simMuonGEMPadDigis
from SimMuon.GEMDigitizer.muonGEMPadDigiClusters_cfi import simMuonGEMPadDigiClusters


def appendGEMChamberMaskerAtReco(process):

    if hasattr(process,'RawToDigi') :

        sys.stderr.write("[appendGEMChamberMasker] : Found RawToDigi, appending filter\n")

        process.simMuonGEMPadDigis = simMuonGEMPadDigis.clone()
        process.simMuonGEMPadDigiClusters = simMuonGEMPadDigiClusters.clone()
        process.simMuonGEMDigis = _gemChamberMasker.clone()
        process.simMuonGEMDigis.digiTag =  cms.InputTag("simMuonGEMDigis", \
                                                        processName = cms.InputTag.skipCurrentProcess())

        process.filteredGEMDigiSequence = cms.Sequence( process.simMuonGEMDigis \
                                                        + process.simMuonGEMPadDigis \
                                                        + process.simMuonGEMPadDigiClusters)

        process.RawToDigi += process.filteredGEMDigiSequence

    return process


