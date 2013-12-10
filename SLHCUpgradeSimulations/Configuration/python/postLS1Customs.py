
import FWCore.ParameterSet.Config as cms

from muonCustoms import customise_csc_PostLS1,customise_csc_hlt


def customisePostLS1(process):

    # deal with CSC separately:
    process = customise_csc_PostLS1(process)

    # all the rest:
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    if hasattr(process,'reconstruction'):
        process=customise_Reco(process)
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
    if hasattr(process,'HLTSchedule'):
        process=customise_HLT(process)
    if hasattr(process,'L1simulation_step'):
        process=customise_L1Emulator(process)
    if hasattr(process,'dqmoffline_step'):
        process=customise_DQM(process)
    if hasattr(process,'dqmHarvesting'):
        process=customise_harvesting(process)
    if hasattr(process,'validation_step'):
        process=customise_Validation(process)

    return process


def digiEventContent(process):
    #extend the event content

    alist=['RAWSIM','FEVTDEBUG','FEVTDEBUGHLT','GENRAW','RAWSIMHLT','FEVT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_simMuonCSCDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_simMuonRPCDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_simHcalUnsuppressedDigis_*_*')

    return process


def customise_DQM(process):
    #process.dqmoffline_step.remove(process.jetMETAnalyzer)
    return process


def customise_Validation(process):
    #process.validation_step.remove(process.PixelTrackingRecHitsValid)
    # We don't run the HLT
    #process.validation_step.remove(process.HLTSusyExoVal)
    #process.validation_step.remove(process.hltHiggsValidator)
    return process


def customise_Digi(process):
    process=digiEventContent(process)
    return process


def customise_L1Emulator(process):
    return process


def customise_RawToDigi(process):
    return process


def customise_DigiToRaw(process):
    return process


def customise_HLT(process):
    process=customise_csc_hlt(process)
    return process


def customise_Reco(process):
    return process


def customise_harvesting(process):
    #process.dqmHarvesting.remove(process.jetMETDQMOfflineClient)
    #process.dqmHarvesting.remove(process.dataCertificationJetMET)
    #process.dqmHarvesting.remove(process.sipixelEDAClient)
    #process.dqmHarvesting.remove(process.sipixelCertification)
    return (process)        

def recoOutputCustoms(process):

    alist=['AODSIM','RECOSIM','FEVTSIM','FEVTDEBUG','FEVTDEBUGHLT','RECODEBUG','RAWRECOSIMHLT','RAWRECODEBUGHLT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_simMuonCSCDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_simMuonRPCDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_simHcalUnsuppressedDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_rawDataCollector_*_*')
    return process

