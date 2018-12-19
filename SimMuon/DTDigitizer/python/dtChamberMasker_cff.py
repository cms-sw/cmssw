from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import sys

from SimMuon.DTDigitizer.dtChamberMasker_cfi import dtChamberMasker as _dtChamberMasker

def appendDTChamberMaskerAtUnpacking(process):

    if hasattr(process,'muonDTDigis') and hasattr(process,'RawToDigi'):
        
        sys.stderr.write("[appendDTChamberMasker] : Found muonDTDigis, applying filter\n")

        process.preDtDigis = process.muonDTDigis.clone()
        process.muonDTDigis = _dtChamberMasker.clone()

        process.muonDTDigis.digiTag = cms.InputTag('preDtDigis') 

        process.RawToDigiTask.add(process.preDtDigis)

        if hasattr(process,"dtTriggerEfficiencyMonitor") :
            process.dtTriggerEfficiencyMonitor.inputTagDDU = 'preDtDigis'

        if hasattr(process,"RandomNumberGeneratorService") :
            process.RandomNumberGeneratorService.muonDTDigis = cms.PSet(
                initialSeed = cms.untracked.uint32(789342)
                )
        else :
            process.RandomNumberGeneratorService = cms.Service(
                "RandomNumberGeneratorService",
                muonDTDigis = cms.PSet(initialSeed = cms.untracked.uint32(789342))
                )

    return process

def appendDTChamberMaskerAtHLT(process):

    if hasattr(process,'hltMuonDTDigis') and \
       ( hasattr(process,'HLTMuonLocalRecoSequence') or \
         hasattr(process,'HLTMuonLocalRecoMeanTimerSequence')) :

        sys.stderr.write("[appendDTChamberMasker] : Found hltMuonDTDigis, applying filter\n")

        process.preHltDtDigis = process.hltMuonDTDigis.clone()
        process.hltMuonDTDigis = _dtChamberMasker.clone()

        process.hltMuonDTDigis.digiTag = "preHltDtDigis"

        process.filteredHltDtDigiSequence = cms.Sequence(process.preHltDtDigis + process.hltMuonDTDigis)
        if hasattr(process,'HLTMuonLocalRecoSequence') :
            print("1")
            process.HLTMuonLocalRecoSequence.replace(process.hltMuonDTDigis, process.filteredHltDtDigiSequence)
        if hasattr(process,'HLTMuonLocalRecoMeanTimerSequence') :
            print("2")
            process.HLTMuonLocalRecoMeanTimerSequence.replace(process.hltMuonDTDigis, process.filteredHltDtDigiSequence)

        process.globalReplace("hltMuonDTDigis",filteredHltDtDigiSequence)

        if hasattr(process,"RandomNumberGeneratorService") :
            process.RandomNumberGeneratorService.hltMuonDTDigis = cms.PSet(
                initialSeed = cms.untracked.uint32(789342)
                )
        else :
            process.RandomNumberGeneratorService = cms.Service(
                "RandomNumberGeneratorService",
                hltMuonDTDigis = cms.PSet(initialSeed = cms.untracked.uint32(789342))
                )
            
    return process

def appendDTChamberMaskerBeforeL1Trigger(process):

    if hasattr(process,'simDtTriggerPrimitiveDigis') and hasattr(process,'SimL1TMuonCommon') :

        sys.stderr.write("[appendDTChamberMasker] : Found simMuonDtTriggerPrimitivesDigis, applying filter\n")

        process.preSimDtTriggerDigis = _dtChamberMasker.clone()

        process.simDtTriggerPrimitiveDigis.digiTag = "preSimDtDigis"

        process.filteredSimDtTriggerSequence = cms.Sequence(process.preSimDtTriggerDigis + process.simDtTriggerPrimitiveDigis)
        process.SimL1TMuonCommon.replace(process.simMuonDTDigis, process.filteredSimDtTriggerSequence)

        if hasattr(process,"RandomNumberGeneratorService") :
            process.RandomNumberGeneratorService.simMuonDTDigis = cms.PSet(
                initialSeed = cms.untracked.uint32(789342)
                )
        else :
            process.RandomNumberGeneratorService = cms.Service(
                "RandomNumberGeneratorService",
                simMuonDTDigis = cms.PSet(initialSeed = cms.untracked.uint32(789342))
                )
            
    return process
