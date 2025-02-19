import FWCore.ParameterSet.Config as cms

trackerSeedValidator = cms.EDAnalyzer("TrackerSeedValidator",
    associators = cms.vstring('quickTrackAssociatorByHits'),
    useFabsEta = cms.bool(False),
    minpT = cms.double(-1),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    min = cms.double(0.0),
    max = cms.double(2.5),
    nintHit = cms.int32(25),
    label = cms.VInputTag(cms.InputTag("newSeedFromTriplets")),
    maxHit = cms.double(24.5),
    TTRHBuilder = cms.string('WithTrackAngle'),
    nintpT = cms.int32(40),
    label_tp_fake = cms.InputTag("cutsTPFake"),
    label_tp_effic = cms.InputTag("cutsTPEffic"),
    useInvPt = cms.bool(False),
    maxpT = cms.double(3),
    outputFile = cms.string(''),
#    outputFile = cms.string('validationPlotsSeed.root'),
    minHit = cms.double(-0.5),
    sim = cms.string('g4SimHits'),
    nint = cms.int32(25),
#the following parameters  are not used at the moment
#but are needed since the seed validator hinerits from multitrack validator base
#to be fixed.
    minPhi = cms.double(-3.15),
    maxPhi = cms.double(3.15),
    nintPhi = cms.int32(36),
    minDxy = cms.double(0),
    maxDxy = cms.double(5),
    nintDxy = cms.int32(50),
    minDz = cms.double(-10),
    maxDz = cms.double(10),
    nintDz = cms.int32(100),
    ptRes_rangeMin = cms.double(-0.1),                                 
    ptRes_rangeMax = cms.double(0.1),
    phiRes_rangeMin = cms.double(-0.003),
    phiRes_rangeMax = cms.double(0.003),
    cotThetaRes_rangeMin = cms.double(-0.01),
    cotThetaRes_rangeMax = cms.double(+0.01),
    dxyRes_rangeMin = cms.double(-0.01),
    dxyRes_rangeMax = cms.double(0.01),
    dzRes_rangeMin = cms.double(-0.05),
    dzRes_rangeMax = cms.double(+0.05),
    ptRes_nbin = cms.int32(100),                                   
    phiRes_nbin = cms.int32(100),                                   
    cotThetaRes_nbin = cms.int32(120),                                   
    dxyRes_nbin = cms.int32(100),                                   
    dzRes_nbin = cms.int32(150),
    useLogPt=cms.untracked.bool(True),
    # TP originating vertical position
    minVertpos = cms.double(0),
    maxVertpos = cms.double(5),
    nintVertpos = cms.int32(100),
    # TP originating z position
    minZpos = cms.double(-10),
    maxZpos = cms.double(10),
    nintZpos = cms.int32(100),
    parametersDefiner = cms.string('LhcParametersDefinerForTP'),
    useGsf=cms.bool(False),
    skipHistoFit=cms.untracked.bool(False)

    
)


