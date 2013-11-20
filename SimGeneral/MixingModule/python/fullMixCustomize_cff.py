import FWCore.ParameterSet.Config as cms

def setCrossingFrameOn(process):

    process.mix.mixObjects.mixCH.crossingFrames = cms.untracked.vstring(
        'CaloHitsTk', 
        'CastorBU', 
        'CastorFI', 
        'CastorPL', 
        'CastorTU', 
        'EcalHitsEB', 
        'EcalHitsEE', 
        'EcalHitsES', 
        'EcalTBH4BeamHits', 
        'HcalHits', 
        'HcalTB06BeamHits', 
        'ZDCHITS')

    process.mix.mixObjects.mixTracks.makeCrossingFrame = cms.untracked.bool(True)
    process.mix.mixObjects.mixVertices.makeCrossingFrame = cms.untracked.bool(True)
    process.mix.mixObjects.mixHepMC.makeCrossingFrame = cms.untracked.bool(True)

    process.mix.mixObjects.mixSH.crossingFrames = cms.untracked.vstring(
        'BSCHits', 
        'FP420SI', 
        'MuonCSCHits', 
        'MuonDTHits', 
        'MuonRPCHits', 
        'TotemHitsRP', 
        'TotemHitsT1', 
        'TotemHitsT2Gem', 
        'TrackerHitsPixelBarrelHighTof', 
        'TrackerHitsPixelBarrelLowTof', 
        'TrackerHitsPixelEndcapHighTof', 
        'TrackerHitsPixelEndcapLowTof', 
        'TrackerHitsTECHighTof', 
        'TrackerHitsTECLowTof', 
        'TrackerHitsTIBHighTof', 
        'TrackerHitsTIBLowTof', 
        'TrackerHitsTIDHighTof', 
        'TrackerHitsTIDLowTof', 
        'TrackerHitsTOBHighTof', 
        'TrackerHitsTOBLowTof')

    return(process)
