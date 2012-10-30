
import FWCore.ParameterSet.Config as cms

from muonCustoms import customise_csc_geom_cond_digi

def digiCustoms(process):
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
   
    #deal with csc
    process=customise_csc_geom_cond_digi(process)

    #extend the event content
    process.FEVTDEBUGoutput.outputCommands.append( 'keep *_simMuonCSCDigis_*_*')
    process.FEVTDEBUGoutput.outputCommands.append( 'keep *_simHcalUnsuppressedDigis_*_*')
                                                   
    
    return process
