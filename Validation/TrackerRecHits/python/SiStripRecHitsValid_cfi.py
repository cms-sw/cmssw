import FWCore.ParameterSet.Config as cms

stripRecHitsValid = cms.EDAnalyzer("SiStripRecHitsValid",
    TopFolderName = cms.string('SiStrip/RecHitsValidation/StiffTrackingRecHits'),

    TH1NumTotrphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(10000.),
        switchon  = cms.bool(True)
   ),

    TH1NumTotStereo = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(10000.),
        switchon  = cms.bool(True)
    ),

    TH1NumTotMatched = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(10000.),
        switchon  = cms.bool(True)

    ),

  TH1Numrphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(5000.),
        switchon  = cms.bool(True)

    ),

  TH1Bunchrphi = cms.PSet( 
        Nbinx          = cms.int32(20),
        xmin           = cms.double(-10.),
        xmax           = cms.double(10.),
        switchon  = cms.bool(True)
    ),

  TH1Eventrphi = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(100.),
        switchon  = cms.bool(True)
    ),

  TH1NumStereo = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(5000.),
        switchon  = cms.bool(True)

    ),

    TH1BunchStereo = cms.PSet( 
        Nbinx          = cms.int32(20),
        xmin           = cms.double(-10.),
        xmax           = cms.double(10.),
        switchon  = cms.bool(True)
    ),

    TH1EventStereo = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(100.),
        switchon  = cms.bool(True)
    ),

  TH1NumMatched = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(5000.),
        switchon  = cms.bool(True)

    ),

    TH1BunchMatched = cms.PSet( 
        Nbinx          = cms.int32(20),
        xmin           = cms.double(-10.),
        xmax           = cms.double(10.),
        switchon  = cms.bool(True)
    ),

    TH1EventMatched = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(100.),
        switchon  = cms.bool(True)
    ),

    TH1Wclusrphi = cms.PSet(
        Nbinx          = cms.int32(20),
        xmin           = cms.double(-0.5),
        xmax           = cms.double(19.5),
        switchon  = cms.bool(True)
    ),

    TH1Adcrphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(500.),#300.
        switchon  = cms.bool(True)
    ),

    TH1Posxrphi = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-6.0),#-6.0
        xmax           = cms.double(+6.0),#+6.0
        switchon  = cms.bool(True)
    ),

    TH1Resolxrphi = cms.PSet( #<error>~20micron 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(0.10),#0.01
        switchon  = cms.bool(True)
    ),

    TH1Resrphi = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-0.10),#-0.02
        xmax           = cms.double(+0.10),#+0.02
        switchon  = cms.bool(True)
    ),

    TH1PullLFrphi = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-10.),#-5.0
        xmax           = cms.double(+10.),#+5.0
        switchon  = cms.bool(True)
    ),

    TH1PullMFrphi = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-10.),#-5.0
        xmax           = cms.double(+10.),#+5.0
        switchon  = cms.bool(True)
    ),

    TH1Chi2rphi = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(50.),
        switchon  = cms.bool(True)
    ),

    TH1NsimHitrphi = cms.PSet( 
        Nbinx          = cms.int32(30),
        xmin           = cms.double(0.),
        xmax           = cms.double(30.),
        switchon  = cms.bool(True)
    ),

    TH1WclusStereo = cms.PSet( 
        Nbinx          = cms.int32(20),
        xmin           = cms.double(-0.5),
        xmax           = cms.double(19.5),
        switchon  = cms.bool(True)
    ),

    TH1AdcStereo = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(500.),#300
        switchon  = cms.bool(True)
    ),

   TH1PosxStereo = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-6.0),
        xmax           = cms.double(+6.0),
        switchon  = cms.bool(True)
    ),

   TH1ResolxStereo = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(0.1),#0.01
        switchon  = cms.bool(True)
    ),

   TH1ResStereo = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-0.10),#-0.02
        xmax           = cms.double(+0.10),#+0.02
        switchon  = cms.bool(True)
    ),

   TH1PullLFStereo = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-4.),
        xmax           = cms.double(+4.),
        switchon  = cms.bool(True)
    ),

   TH1PullMFStereo = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-10.),#-4.0
        xmax           = cms.double(+10.),#+4.0
        switchon  = cms.bool(True)
    ),

   TH1Chi2Stereo = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(100.),#50
        switchon  = cms.bool(True)
    ),

    TH1NsimHitStereo = cms.PSet( 
        Nbinx          = cms.int32(30),
        xmin           = cms.double(0.),
        xmax           = cms.double(30.),
        switchon  = cms.bool(True)
    ),

   TH1PosxMatched = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-10.0),#-6.0
        xmax           = cms.double(+10.0),#+6.0
        switchon  = cms.bool(True)
    ),

   TH1PosyMatched = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-10.0),#-6.0
        xmax           = cms.double(+10.0),#+6.0
        switchon  = cms.bool(True)
    ),

   TH1ResolxMatched = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(0.1),#0.01
        switchon  = cms.bool(True)
    ),

   TH1ResolyMatched = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(0.25),#0.05
        switchon  = cms.bool(True)
    ),

   TH1ResxMatched = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-0.1),#-0.02
        xmax           = cms.double(+0.1),#+0.02
        switchon  = cms.bool(True)
    ),

   TH1ResyMatched = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-10.), #-1
        xmax           = cms.double(+10. ), #+1
        switchon  = cms.bool(True) 
    ),

   TH1Chi2Matched = cms.PSet( 
        Nbinx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(50.),
        switchon  = cms.bool(True)
    ),

    TH1NsimHitMatched = cms.PSet( 
        Nbinx          = cms.int32(30),
        xmin           = cms.double(0.),
        xmax           = cms.double(30.),
        switchon  = cms.bool(True)
    ),

    SubDetList = cms.vstring('TIB','TOB','TID','TEC'),
    associatePixel = cms.bool(False),
    stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    ROUList = cms.vstring('g4SimHitsTrackerHitsTIBLowTof', 
        'g4SimHitsTrackerHitsTIBHighTof', 
        'g4SimHitsTrackerHitsTIDLowTof', 
        'g4SimHitsTrackerHitsTIDHighTof', 
        'g4SimHitsTrackerHitsTOBLowTof', 
        'g4SimHitsTrackerHitsTOBHighTof', 
        'g4SimHitsTrackerHitsTECLowTof', 
        'g4SimHitsTrackerHitsTECHighTof'),
    associateRecoTracks = cms.bool(False),
    associateStrip = cms.bool(True),
    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    RecHitProducer = cms.string('siStripMatchedRecHits'),
    pixelSimLinkSrc = cms.InputTag("simSiPixelDigis"),
    stripSimLinkSrc = cms.InputTag("simSiStripDigis"),
    verbose = cms.untracked.bool(False)
)


