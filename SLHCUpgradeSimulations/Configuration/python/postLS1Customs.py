
import FWCore.ParameterSet.Config as cms

from muonCustoms import customise_csc_geom_cond_digi

def turnOffXFrame(process):
    #turn off the crossing frame
    process.mix.mixObjects.mixSH.crossingFrames = cms.untracked.vstring(
        'BSCHits',
        'FP420SI',
        'MuonCSCHits',
        'MuonDTHits',
        'MuonRPCHits',
        'TotemHitsRP',
        'TotemHitsT1',
        'TotemHitsT2Gem')
    process.mix.mixObjects.mixCH.crossingFrames = cms.untracked.vstring('')
    process.mix.mixObjects.mixTracks.makeCrossingFrame = cms.untracked.bool(False)
    process.mix.mixObjects.mixVertices.makeCrossingFrame = cms.untracked.bool(False)
    process.mix.mixObjects.mixHepMC.makeCrossingFrame = cms.untracked.bool(False)
    process.digitisation_step.remove(process.simSiStripDigiSimLink)
    process.digitisation_step.remove(process.mergedtruth)
    return process

def digiEventContent(process):
    #extend the event content
    if hasattr(process,'FEVTDEBUGoutput'):
        process.FEVTDEBUGoutput.outputCommands.append( 'keep *_simMuonCSCDigis_*_*')
        process.FEVTDEBUGoutput.outputCommands.append( 'keep *_simMuonRPCDigis_*_*')
        process.FEVTDEBUGoutput.outputCommands.append( 'keep *_simHcalUnsuppressedDigis_*_*')
    if hasattr(process,'GENRAWoutput'):
        print 'ok'
        process.GENRAWoutput.outputCommands.append( 'keep *_simMuonCSCDigis_*_*')
        process.GENRAWoutput.outputCommands.append( 'keep *_simMuonRPCDigis_*_*')
        process.GENRAWoutput.outputCommands.append( 'keep *_simHcalUnsuppressedDigis_*_*')

    return process    

def digiCustomsRelVal(process):
    #deal with csc
    process=customise_csc_geom_cond_digi(process)
    process=digiEventContent(process)
    process.digi2raw_step.remove(process.cscpacker)
    return process

def digiCustoms(process):
    process=turnOffXFrame(process)
    process=digiCustomsRelVal(process)
    return process


def hltCustoms(process):
    process.CSCGeometryESModule.useGangedStripsInME1a = False

    process.hltCsc2DRecHits.readBadChannels = cms.bool(False)
    process.hltCsc2DRecHits.CSCUseTimingCorrections = cms.bool(False)
    process.hltCsc2DRecHits.CSCUseGasGainCorrection = cms.bool(False)

    # Switch input for CSCRecHitD to  s i m u l a t e d  digis

    process.hltCsc2DRecHits.wireDigiTag  = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
    process.hltCsc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")

    return process

def recoCustoms(process):

    # ME1/1A is  u n g a n g e d  Post-LS1

    process.CSCGeometryESModule.useGangedStripsInME1a = False

    # Turn off some flags for CSCRecHitD that are turned ON in default config

    process.csc2DRecHits.readBadChannels = cms.bool(False)
    process.csc2DRecHits.CSCUseTimingCorrections = cms.bool(False)
    process.csc2DRecHits.CSCUseGasGainCorrection = cms.bool(False)

    # Switch input for CSCRecHitD to  s i m u l a t e d  digis

    process.csc2DRecHits.wireDigiTag  = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
    process.csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")

    return process

