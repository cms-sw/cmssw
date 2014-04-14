import FWCore.ParameterSet.Config as cms

def customise(process):
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
    if hasattr(process,'L1simulation_step'):
       process=customise_L1Emulator(process)
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    if hasattr(process,'reconstruction'):
        process=customise_Reco(process)
    if hasattr(process,'dqmoffline_step'):
        process=customise_DQM(process)
    if hasattr(process,'dqmHarvesting'):
        process=customise_harvesting(process)
    if hasattr(process,'validation_step'):
        process=customise_Validation(process)
    return process

def customise_Digi(process):
    return process

def customise_L1Emulator(process):
    process.simCscTriggerPrimitiveDigis.rpcDigiProducer =  cms.untracked.InputTag("simMuonRPCDigis","")
    tmb = process.simCscTriggerPrimitiveDigis.tmbSLHC
    tmb.me3141ILT = cms.PSet(
        runME3141ILT = cms.untracked.bool(True),
        debugRPCMatching = cms.untracked.bool(True),
        maxDeltaBXRPC = cms.untracked.int32(0),
        maxDeltaRollRPC = cms.untracked.int32(0),
        maxDeltaStripRPC = cms.untracked.int32(1),
        dropLowQualityCLCTsNoRPC = cms.untracked.bool(True),
        dropLowQualityALCTsNoRPCs = cms.untracked.bool(True),
    )
    if tmb.me11ILT.runME11ILT:
        process.simCscTriggerPrimitiveDigis.clctSLHC.clctNplanesHitPattern = 3
        process.simCscTriggerPrimitiveDigis.clctSLHC.clctPidThreshPretrig = 2
        process.simCscTriggerPrimitiveDigis.clctParam07.clctPidThreshPretrig = 2

    return process

def customise_DigiToRaw(process):
    return process

def customise_RawToDigi(process):
    return process

def customise_Reco(process):
    return process

def customise_DQM(process):
    return process

def customise_Validation(process):
    return process

def customise_harvesting(process):
    return (process)
