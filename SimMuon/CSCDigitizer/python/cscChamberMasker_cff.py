import FWCore.ParameterSet.Config as cms
import sys

from SimMuon.CSCDigitizer.cscChamberMasker_cfi import cscChamberMasker as _cscChamberMasker

def appendCSCChamberMaskerAtUnpacking(process):

    if hasattr(process,'muonCSCDigis') :
        
        sys.stderr.write("[appendCSCChamberMasker] : Found muonCSCDigis, applying filter\n")

        process.preCSCDigis = process.muonCSCDigis.clone()
        process.muonCSCDigis = _cscChamberMasker.clone()

        process.muonCSCDigis.stripDigiTag = cms.InputTag("preCSCDigis", "MuonCSCStripDigi")
        process.muonCSCDigis.wireDigiTag = cms.InputTag("preCSCDigis", "MuonCSCWireDigi") 
        process.muonCSCDigis.comparatorDigiTag = cms.InputTag("preCSCDigis", "MuonCSCComparatorDigi")
        process.muonCSCDigis.rpcDigiTag = cms.InputTag("preCSCDigis", "MuonCSCRPCDigi") 
        process.muonCSCDigis.alctDigiTag = cms.InputTag("preCSCDigis", "MuonCSCALCTDigi") 
        process.muonCSCDigis.clctDigiTag = cms.InputTag("preCSCDigis", "MuonCSCCLCTDigi") 

        process.RawToDigiTask.add(process.preCSCDigis)

        if hasattr(process,"RandomNumberGeneratorService") :
            process.RandomNumberGeneratorService.muonCSCDigis = cms.PSet(
                initialSeed = cms.untracked.uint32(789342)
                )
        else :
            process.RandomNumberGeneratorService = cms.Service(
                "RandomNumberGeneratorService",
                muonCSCDigis = cms.PSet(initialSeed = cms.untracked.uint32(789342))
                )

    return process

def appendCSCChamberMaskerAtHLT(process):

    if hasattr(process,'hltMuonCSCDigis') :

        sys.stderr.write("[appendCSCChamberMasker] : Found hltMuonCSCDigis, applying filter\n")

        process.preHltCSCDigis = process.hltMuonCSCDigis.clone()
        process.hltMuonCSCDigis = _cscChamberMasker.clone()

        process.hltMuonCSCDigis.stripDigiTag = cms.InputTag("preHltCSCDigis", "MuonCSCStripDigi")
        process.hltMuonCSCDigis.wireDigiTag = cms.InputTag("preHltCSCDigis", "MuonCSCWireDigi") 
        process.hltMuonCSCDigis.comparatorDigiTag = cms.InputTag("preHltCSCDigis", "MuonCSCComparatorDigi")
        process.hltMuonCSCDigis.rpcDigiTag = cms.InputTag("preHltCSCDigis", "MuonCSCRPCDigi") 
        process.hltMuonCSCDigis.alctDigiTag = cms.InputTag("preHltCSCDigis", "MuonCSCALCTDigi") 
        process.hltMuonCSCDigis.clctDigiTag = cms.InputTag("preHltCSCDigis", "MuonCSCCLCTDigi") 


        process.filteredHltCSCDigiSequence = cms.Sequence(process.preHltCSCDigis + process.hltMuonCSCDigis)
        process.HLTMuonLocalRecoSequence.replace(process.hltMuonCSCDigis, process.filteredHltCSCDigiSequence)

        if hasattr(process,"RandomNumberGeneratorService") :
            process.RandomNumberGeneratorService.hltMuonCSCDigis = cms.PSet(
                initialSeed = cms.untracked.uint32(789342)
                )
        else :
            process.RandomNumberGeneratorService = cms.Service(
                "RandomNumberGeneratorService",
                hltMuonCSCDigis = cms.PSet(initialSeed = cms.untracked.uint32(789342))
                )
            
    return process
