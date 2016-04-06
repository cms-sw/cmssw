import FWCore.ParameterSet.Config as cms

def customise2019(process):
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
    if hasattr(process,'L1simulation_step'):
        process=customise_L1Emulator2019(process,'pt0')
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
    if hasattr(process,'HLTSchedule'):
        process=customise_gem_hlt(process)
    return process

def customise_Digi(process):
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
    return process

def outputCustoms(process):
    return process

def customise_gem_hlt(process):
    process.hltL2OfflineMuonSeeds.EnableGEMMeasurement = cms.bool( True )
    process.hltL2Muons.L2TrajBuilderParameters.EnableGEMMeasurement = cms.bool( True )
    process.hltL2Muons.BWFilterParameters.EnableGEMMeasurement = cms.bool( True )
    return process


