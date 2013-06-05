
import FWCore.ParameterSet.Config as cms

from muonCustoms import customise_csc_geom_cond_digi,digitizer_timing_pre3_median,unganged_me1a_geometry

def customisePostLS1(process):
    #move this first one to the geometry
    process=unganged_me1a_geometry(process)
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    if hasattr(process,'reconstruction'):
        process=customise_Reco(process)
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
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
    process.dqmoffline_step.remove(process.muonAnalyzer)
    process.dqmoffline_step.remove(process.jetMETAnalyzer)
    return process

def customise_Validation(process):
    process.validation_step.remove(process.PixelTrackingRecHitsValid)
    # We don't run the HLT
    process.validation_step.remove(process.HLTSusyExoVal)
    process.validation_step.remove(process.hltHiggsValidator)
    process.validation_step.remove(process.relvalMuonBits)
    return process

def customise_Digi(process):
    #deal with csc
    process=digitizer_timing_pre3_median(process)
    process=digiEventContent(process)
    process.CSCIndexerESProducer.AlgoName=cms.string("CSCIndexerPostls1")
    process.CSCChannelMapperESProducer.AlgoName=cms.string("CSCChannelMapperPostls1")
    return process

def customise_RawToDigi(process):
    return process

def customise_DigiToRaw(process):
    process.digi2raw_step.remove(process.cscpacker)
    return process


def customise_HLT(process):
    process.CSCGeometryESModule.useGangedStripsInME1a = False

    process.hltCsc2DRecHits.readBadChannels = cms.bool(False)
    process.hltCsc2DRecHits.CSCUseGasGainCorrection = cms.bool(False)

    # Switch input for CSCRecHitD to  s i m u l a t e d  digis

    process.hltCsc2DRecHits.wireDigiTag  = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
    process.hltCsc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")

    return process

def customise_Reco(process):

    # ME1/1A is  u n g a n g e d  Post-LS1

    process.CSCGeometryESModule.useGangedStripsInME1a = False

    # Turn off some flags for CSCRecHitD that are turned ON in default config

    process.csc2DRecHits.readBadChannels = cms.bool(False)
    process.csc2DRecHits.CSCUseGasGainCorrection = cms.bool(False)

    # Switch input for CSCRecHitD to  s i m u l a t e d  digis

    process.csc2DRecHits.wireDigiTag  = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
    process.csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")

    process.CSCIndexerESProducer.AlgoName=cms.string("CSCIndexerPostls1")
    process.CSCChannelMapperESProducer.AlgoName=cms.string("CSCChannelMapperPostls1")

    return process

def customise_harvesting(process):
    process.dqmHarvesting.remove(process.jetMETDQMOfflineClient)
    process.dqmHarvesting.remove(process.dataCertificationJetMET)
    process.dqmHarvesting.remove(process.sipixelEDAClient)
    process.dqmHarvesting.remove(process.sipixelCertification)
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

            
