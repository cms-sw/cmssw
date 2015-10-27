import FWCore.ParameterSet.Config as cms

mixSimHits = cms.PSet(
    input = cms.VInputTag(  # note that this list needs to be in the same order as the subdets
        #cms.InputTag("g4SimHits","BSCHits"), cms.InputTag("g4SimHits","BCM1FHits"), cms.InputTag("g4SimHits","PLTHits"), cms.InputTag("g4SimHits","FP420SI"),
        cms.InputTag("g4SimHits","MuonCSCHits"), cms.InputTag("g4SimHits","MuonDTHits"), cms.InputTag("g4SimHits","MuonRPCHits"), 
        #cms.InputTag("g4SimHits","TotemHitsRP"), cms.InputTag("g4SimHits","TotemHitsT1"), cms.InputTag("g4SimHits","TotemHitsT2Gem"),
        cms.InputTag("g4SimHits","TrackerHitsPixelBarrelHighTof"), cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"), 
        cms.InputTag("g4SimHits","TrackerHitsPixelEndcapHighTof"), cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"), cms.InputTag("g4SimHits","TrackerHitsTECHighTof"), cms.InputTag("g4SimHits","TrackerHitsTECLowTof"), cms.InputTag("g4SimHits","TrackerHitsTIBHighTof"), 
        cms.InputTag("g4SimHits","TrackerHitsTIBLowTof"), cms.InputTag("g4SimHits","TrackerHitsTIDHighTof"), cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"), cms.InputTag("g4SimHits","TrackerHitsTOBHighTof"), cms.InputTag("g4SimHits","TrackerHitsTOBLowTof")),
    type = cms.string('PSimHit'),
    subdets = cms.vstring(
       # 'BSCHits', 
       # 'BCM1FHits',
       # 'PLTHits',
       # 'FP420SI', 
        'MuonCSCHits', 
        'MuonDTHits', 
        'MuonRPCHits', 
       # 'TotemHitsRP', 
       # 'TotemHitsT1', 
       # 'TotemHitsT2Gem', 
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
        'TrackerHitsTOBLowTof'),
    crossingFrames = cms.untracked.vstring(
        'MuonCSCHits', 
        'MuonDTHits', 
        'MuonRPCHits'), 
    #crossingFrames = cms.untracked.vstring(
    #    'BSCHits',
    #    'BCM1FHits',
    #    'PLTHits'
    #    'FP420SI',
    #    'MuonCSCHits',
    #    'MuonDTHits',
    #    'MuonRPCHits',
    #    'TotemHitsRP',
    #    'TotemHitsT1',
    #    'TotemHitsT2Gem')
)
mixCaloHits = cms.PSet(
    input = cms.VInputTag(  # note that this list needs to be in the same order as the subdets
        #cms.InputTag("g4SimHits","CaloHitsTk"), cms.InputTag("g4SimHits","CastorBU"), cms.InputTag("g4SimHits","CastorPL"), cms.InputTag("g4SimHits","CastorTU"), 
        cms.InputTag("g4SimHits","CastorFI"),
        cms.InputTag("g4SimHits","EcalHitsEB"), cms.InputTag("g4SimHits","EcalHitsEE"), cms.InputTag("g4SimHits","EcalHitsES"),
        #cms.InputTag("g4SimHits","EcalTBH4BeamHits"), cms.InputTag("g4SimHits","HcalTB06BeamHits"),
        cms.InputTag("g4SimHits","HcalHits"), 
        cms.InputTag("g4SimHits","ZDCHITS")),
    type = cms.string('PCaloHit'),
    subdets = cms.vstring(
        #'CaloHitsTk', 
        #'CastorBU', 
        'CastorFI', 
        #'CastorPL', 
        #'CastorTU', 
        'EcalHitsEB', 
        'EcalHitsEE', 
        'EcalHitsES', 
        #'EcalTBH4BeamHits', 
        'HcalHits', 
        #'HcalTB06BeamHits', 
        'ZDCHITS'),
    crossingFrames = cms.untracked.vstring()
)
mixSimTracks = cms.PSet(
    makeCrossingFrame = cms.untracked.bool(False),
    input = cms.VInputTag(cms.InputTag("g4SimHits")),
    type = cms.string('SimTrack')
)
mixSimVertices = cms.PSet(
    makeCrossingFrame = cms.untracked.bool(False),
    input = cms.VInputTag(cms.InputTag("g4SimHits")),
    type = cms.string('SimVertex')
)
mixHepMCProducts = cms.PSet(
    makeCrossingFrame = cms.untracked.bool(False),
    input = cms.VInputTag(cms.InputTag("generatorSmeared"),cms.InputTag("generator")),
    type = cms.string('HepMCProduct')
)

theMixObjects = cms.PSet(
    mixCH = cms.PSet(
        mixCaloHits
    ),
    mixTracks = cms.PSet(
        mixSimTracks
    ),
    mixVertices = cms.PSet(
        mixSimVertices
    ),
    mixSH = cms.PSet(
        mixSimHits
    ),
    mixHepMC = cms.PSet(
        mixHepMCProducts
    )
)


mixPCFSimHits = cms.PSet(
    input = cms.VInputTag(cms.InputTag("CFWriter","g4SimHitsBSCHits"), cms.InputTag("CFWriter","g4SimHitsBCM1FHits"), cms.InputTag("CFWriter","g4SimHitsPLTHits"), cms.InputTag("CFWriter","g4SimHitsFP420SI"), cms.InputTag("CFWriter","g4SimHitsMuonCSCHits"), cms.InputTag("CFWriter","g4SimHitsMuonDTHits"), cms.InputTag("CFWriter","g4SimHitsMuonRPCHits"), 
        cms.InputTag("CFWriter","g4SimHitsTotemHitsRP"), cms.InputTag("CFWriter","g4SimHitsTotemHitsT1"), cms.InputTag("CFWriter","g4SimHitsTotemHitsT2Gem"), cms.InputTag("CFWriter","g4SimHitsTrackerHitsPixelBarrelHighTof"), cms.InputTag("CFWriter","g4SimHitsTrackerHitsPixelBarrelLowTof"), 
        cms.InputTag("CFWriter","g4SimHitsTrackerHitsPixelEndcapHighTof"), cms.InputTag("CFWriter","g4SimHitsTrackerHitsPixelEndcapLowTof"), cms.InputTag("CFWriter","g4SimHitsTrackerHitsTECHighTof"), cms.InputTag("CFWriter","g4SimHitsTrackerHitsTECLowTof"), cms.InputTag("CFWriter","g4SimHitsTrackerHitsTIBHighTof"), 
        cms.InputTag("CFWriter","g4SimHitsTrackerHitsTIBLowTof"), cms.InputTag("CFWriter","g4SimHitsTrackerHitsTIDHighTof"), cms.InputTag("CFWriter","g4SimHitsTrackerHitsTIDLowTof"), cms.InputTag("CFWriter","g4SimHitsTrackerHitsTOBHighTof"), cms.InputTag("CFWriter","g4SimHitsTrackerHitsTOBLowTof")),
    type = cms.string('PSimHitPCrossingFrame'),
    subdets = cms.vstring('BSCHits', 
        'BCM1FHits',
        'PLTHits',
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
)

mixPCFCaloHits = cms.PSet(
    input = cms.VInputTag(cms.InputTag("CFWriter","g4SimHitsCaloHitsTk"), cms.InputTag("CFWriter","g4SimHitsCastorBU"), cms.InputTag("CFWriter","g4SimHitsCastorFI"), cms.InputTag("CFWriter","g4SimHitsCastorPL"), cms.InputTag("CFWriter","g4SimHitsCastorTU"), 
        cms.InputTag("CFWriter","g4SimHitsEcalHitsEB"), cms.InputTag("CFWriter","g4SimHitsEcalHitsEE"), cms.InputTag("CFWriter","g4SimHitsEcalHitsES"), cms.InputTag("CFWriter","g4SimHitsEcalTBH4BeamHits"), cms.InputTag("CFWriter","g4SimHitsHcalHits"), 
        cms.InputTag("CFWriter","g4SimHitsHcalTB06BeamHits"), cms.InputTag("CFWriter","g4SimHitsZDCHITS")),
    type = cms.string('PCaloHitPCrossingFrame'),
    subdets = cms.vstring('CaloHitsTk', 
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
)

mixPCFSimTracks = cms.PSet(
    input = cms.VInputTag(cms.InputTag("CFWriter","g4SimHits")),
    type = cms.string('SimTrackPCrossingFrame')
)

mixPCFSimVertices = cms.PSet(
    input = cms.VInputTag(cms.InputTag("CFWriter","g4SimHits")),
    type = cms.string('SimVertexPCrossingFrame')
)

mixPCFHepMCProducts = cms.PSet(
    input = cms.VInputTag(cms.InputTag("CFWriter","generator")),
    type = cms.string('HepMCProductPCrossingFrame')
)
