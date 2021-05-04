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
        
    from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
    run2_GEM_2017.toModify( process.mix.mixObjects,
        mixSH = dict(
            crossingFrames = process.mix.mixObjects.mixSH.crossingFrames + [ 'MuonGEMHits' ]
        )
    )
    from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
    run3_GEM.toModify( process.mix.mixObjects,
        mixSH = dict(
            crossingFrames = process.mix.mixObjects.mixSH.crossingFrames + [ 'MuonGEMHits' ]
        )
    )
    from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
    phase2_muon.toModify( process.mix.mixObjects,
        mixSH = dict(
            crossingFrames = process.mix.mixObjects.mixSH.crossingFrames + [ 'MuonME0Hits' ]
        )
    )
    from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
    phase2_timing_layer.toModify( process.mix.mixObjects,
        mixSH = dict(
            crossingFrames = process.mix.mixObjects.mixSH.crossingFrames + [ 'FastTimerHitsBarrel', 'FastTimerHitsEndcap' ]
        )
    )

    return(process)
