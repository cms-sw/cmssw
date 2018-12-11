import FWCore.ParameterSet.Config as cms
import sys

from SimMuon.RPCDigitizer.rpcChamberMasker_cfi import rpcChamberMasker as _rpcChamberMasker
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon

def appendRPCChamberMaskerAtReco(process):

    if phase2_muon.isChosen():
        appendRPCChamberMaskerBeforeRecHits(process)
    else :
        appendRPCChamberMaskerAtUnpacking(process)

    return process


# To be used for PhaseII
# no packin/unpacking is available for RE3/1 RE4/1
# must start from sim digis  
def appendRPCChamberMaskerBeforeRecHits(process):

    if hasattr(process,'rpcRecHits') :

        sys.stderr.write("[appendRPCChamberMasker] : Found rpcRecHits, applying filter\n")

        process.rpcAgedDigis = _rpcChamberMasker.clone()
        process.rpcAgedDigis.digiTag = cms.InputTag('simMuonRPCDigis')

        process.rpcRecHits = process.rpcRecHits.clone()
        process.rpcRecHits.rpcDigiLabel = cms.InputTag('rpcAgedDigis')

        process.filteredRpcDigiSequence = cms.Sequence(process.rpcAgedDigis \
                                                       + process.rpcRecHits)

        process.reconstruction.replace(process.rpcRecHits, \
                                       process.filteredRpcDigiSequence)

        if hasattr(process,"RandomNumberGeneratorService") :
            process.RandomNumberGeneratorService.rpcAgedDigis = cms.PSet(
                initialSeed = cms.untracked.uint32(789342)
                )
        else :
            process.RandomNumberGeneratorService = cms.Service(
                "RandomNumberGeneratorService",
                rpcAgedDigis = cms.PSet(initialSeed = cms.untracked.uint32(789342))
                )

        return process


# To be used for PhaseII
# exploit unpacking of digis  
def appendRPCChamberMaskerAtUnpacking(process):

    if hasattr(process,'muonRPCDigis') :

        sys.stderr.write("[appendRPCChamberMasker] : Found muonRPCDigis, applying filter\n")

        process.preRPCDigis = process.muonRPCDigis.clone()
        process.muonRPCDigis = _rpcChamberMasker.clone()

        if hasattr(process,"RandomNumberGeneratorService") :
            process.RandomNumberGeneratorService.muonRPCDigis = cms.PSet(
                initialSeed = cms.untracked.uint32(789342)
                )
        else :
            process.RandomNumberGeneratorService = cms.Service(
                "RandomNumberGeneratorService",
                muonRPCDigis = cms.PSet(initialSeed = cms.untracked.uint32(789342))
                )

        process.RawToDigiTask.add(process.preRPCDigis)

    return process
