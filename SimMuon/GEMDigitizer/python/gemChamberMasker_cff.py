import FWCore.ParameterSet.Config as cms
import sys

from SimMuon.GEMDigitizer.gemChamberMasker_cfi import gemChamberMasker as _gemChamberMasker


def appendGEMChamberMaskerAtReco(process):

    if hasattr(process,'RawToDigi') :

        sys.stderr.write("[appendGEMChamberMasker] : Found RawToDigi, appending filter\n")

        process.simMuonGEMDigis = _gemChamberMasker.clone()
        process.simMuonGEMDigis.digiTag =  cms.InputTag("simMuonGEMDigis", \
                                                        processName = cms.InputTag.skipCurrentProcess())

        process.filteredGEMDigiSequence = cms.Sequence( process.simMuonGEMDigis)

        process.RawToDigi += process.filteredGEMDigiSequence

    return process
