import FWCore.ParameterSet.Config as cms

def customiseDefault(process):
    if hasattr(process,'pfTrack'):
        process.pfTrack.TrajInEvents = cms.bool(True)
    if hasattr(process,'csc2DRecHits'):
        process.csc2DRecHits.readBadChannels = cms.bool(False)

    if hasattr(process,'validation_step'):
        process.validation_step.remove(process.HLTSusyExoValFastSim)
        process.validation_step.remove(process.hltHiggsValidator)

    process.trackerNumberingSLHCGeometry.layerNumberPXB = cms.uint32(20)
    process.trackerTopologyConstants.pxb_layerStartBit = cms.uint32(20)
    process.trackerTopologyConstants.pxb_ladderStartBit = cms.uint32(12)
    process.trackerTopologyConstants.pxb_moduleStartBit = cms.uint32(2)
    process.trackerTopologyConstants.pxb_layerMask = cms.uint32(15)
    process.trackerTopologyConstants.pxb_ladderMask = cms.uint32(255)
    process.trackerTopologyConstants.pxb_moduleMask = cms.uint32(1023)
    process.trackerTopologyConstants.pxf_diskStartBit = cms.uint32(18)
    process.trackerTopologyConstants.pxf_bladeStartBit = cms.uint32(12)
    process.trackerTopologyConstants.pxf_panelStartBit = cms.uint32(10)
    process.trackerTopologyConstants.pxf_moduleMask = cms.uint32(255)
    return process

def customisePhase2(process):
    process=customiseDefault(process)
    process.famosSimHits.MaterialEffects.PairProduction = cms.bool(False)
    process.famosSimHits.MaterialEffects.Bremsstrahlung = cms.bool(False)
    process.famosSimHits.MaterialEffects.MuonBremsstrahlung = cms.bool(False)
    process.famosSimHits.MaterialEffects.EnergyLoss = cms.bool(False)
    process.famosSimHits.MaterialEffects.MultipleScattering = cms.bool(False)
    # keep NI so to allow thickness to be properly treated in the interaction geometry
    process.famosSimHits.MaterialEffects.NuclearInteraction = cms.bool(True)
    process.KFFittingSmootherWithOutlierRejection.EstimateCut = cms.double(50.0)

    return process

