import FWCore.ParameterSet.Config as cms

trackAssociatorByHits = cms.EDProducer("TrackAssociatorByHitsProducer",
    Quality_SimToReco = cms.double(0.5),
    associateRecoTracks = cms.bool(True),
    UseGrouped = cms.bool(True),
    associatePixel = cms.bool(True),
    ROUList = cms.vstring('TrackerHitsTIBLowTof', 
        'TrackerHitsTIBHighTof', 
        'TrackerHitsTIDLowTof', 
        'TrackerHitsTIDHighTof', 
        'TrackerHitsTOBLowTof', 
        'TrackerHitsTOBHighTof', 
        'TrackerHitsTECLowTof', 
        'TrackerHitsTECHighTof', 
        'TrackerHitsPixelBarrelLowTof', 
        'TrackerHitsPixelBarrelHighTof', 
        'TrackerHitsPixelEndcapLowTof', 
        'TrackerHitsPixelEndcapHighTof'),
    UseSplitting = cms.bool(True),
    UsePixels = cms.bool(True),
    ThreeHitTracksAreSpecial = cms.bool(True),
    AbsoluteNumberOfHits = cms.bool(False),
    associateStrip = cms.bool(True),
    pixelSimLinkSrc = cms.InputTag("simSiPixelDigis"),
    stripSimLinkSrc = cms.InputTag("simSiStripDigis"),
    Purity_SimToReco = cms.double(0.75),
    Cut_RecoToSim = cms.double(0.75),
    SimToRecoDenominator = cms.string('sim'), ##"reco"
    simHitTpMapTag = cms.InputTag("simHitTPAssocProducer")
 )


from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(trackAssociatorByHits,
    pixelSimLinkSrc = "mixData:PixelDigiSimLink",
    stripSimLinkSrc = "mixData:StripDigiSimLink",
)
