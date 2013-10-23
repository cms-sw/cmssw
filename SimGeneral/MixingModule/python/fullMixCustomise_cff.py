import FWCore.ParameterSet.Config as cms

def customise(process):

    process.mix.mixObjects.mixCH.input = cms.VInputTag(cms.InputTag("g4SimHits","CaloHitsTk"),
         cms.InputTag("g4SimHits","CastorBU"),
         cms.InputTag("g4SimHits","CastorFI"),
         cms.InputTag("g4SimHits","CastorPL"),
         cms.InputTag("g4SimHits","CastorTU"),
         cms.InputTag("g4SimHits","EcalHitsEB"),
         cms.InputTag("g4SimHits","EcalHitsEE"),
         cms.InputTag("g4SimHits","EcalHitsES"),
         cms.InputTag("g4SimHits","EcalTBH4BeamHits"),
         cms.InputTag("g4SimHits","HcalHits"),
         cms.InputTag("g4SimHits","HcalTB06BeamHits"),
         cms.InputTag("g4SimHits","ZDCHITS"))
    
    process.mix.mixObjects.mixCH.subdets = cms.vstring(
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

    process.mix.mixObjects.mixSH.input = cms.VInputTag(cms.InputTag("g4SimHits","BSCHits"),
        cms.InputTag("g4SimHits","FP420SI"),
        cms.InputTag("g4SimHits","MuonCSCHits"),
        cms.InputTag("g4SimHits","MuonDTHits"),
        cms.InputTag("g4SimHits","MuonRPCHits"),
        cms.InputTag("g4SimHits","TotemHitsRP"),
        cms.InputTag("g4SimHits","TotemHitsT1"),
        cms.InputTag("g4SimHits","TotemHitsT2Gem"),
        cms.InputTag("g4SimHits","TrackerHitsPixelBarrelHighTof"),
        cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
        cms.InputTag("g4SimHits","TrackerHitsPixelEndcapHighTof"),
        cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"),
        cms.InputTag("g4SimHits","TrackerHitsTECHighTof"),
        cms.InputTag("g4SimHits","TrackerHitsTECLowTof"),
        cms.InputTag("g4SimHits","TrackerHitsTIBHighTof"),
        cms.InputTag("g4SimHits","TrackerHitsTIBLowTof"),
        cms.InputTag("g4SimHits","TrackerHitsTIDHighTof"),
        cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"),
        cms.InputTag("g4SimHits","TrackerHitsTOBHighTof"),
        cms.InputTag("g4SimHits","TrackerHitsTOBLowTof"))
    
    process.mix.mixObjects.mixSH.subdets = cms.vstring(
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
