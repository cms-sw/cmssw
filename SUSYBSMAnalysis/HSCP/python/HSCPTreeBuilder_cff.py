import FWCore.ParameterSet.Config as cms

dedxHarm2 = cms.EDProducer("DeDxEstimatorProducer",
    tracks                     = cms.InputTag("TrackRefitter"),
    trajectoryTrackAssociation = cms.InputTag("TrackRefitter"),

    estimator      = cms.string('generic'),
    exponent       = cms.double(-2.0),

    UseStrip       = cms.bool(True),
    UsePixel       = cms.bool(True),
    MeVperADCStrip = cms.double(3.61e-06*250),
    MeVperADCPixel = cms.double(3.61e-06),

    MisCalib_Mean      = cms.untracked.double(1.0),
    MisCalib_Sigma     = cms.untracked.double(0.00),

    UseCalibration  = cms.bool(False),
    calibrationPath = cms.string(""),
)

dedxNPHarm2             = dedxHarm2.clone()
dedxNPHarm2.UsePixel    = cms.bool(False)

dedxCHarm2                = dedxHarm2.clone()
dedxCHarm2.UseCalibration = cms.bool(True)
dedxCHarm2.calibrationPath = cms.string("file:Gains.root")

dedxCNPHarm2                = dedxNPHarm2.clone()
dedxCNPHarm2.UseCalibration = cms.bool(True)
dedxCNPHarm2.calibrationPath = cms.string("file:Gains.root")

####################################################################################

dedxProd               = cms.EDProducer("DeDxDiscriminatorProducer",
    tracks                     = cms.InputTag("TrackRefitter"),
    trajectoryTrackAssociation = cms.InputTag("TrackRefitter"),

    Reccord            = cms.untracked.string("SiStripDeDxMip_3D_Rcd"),
    Formula            = cms.untracked.uint32(0),
#    ProbabilityMode    = cms.untracked.string("Integral"),
    ProbabilityMode    = cms.untracked.string("Accumulation"),


    UseStrip           = cms.bool(True),
    UsePixel           = cms.bool(True),
    MeVperADCStrip     = cms.double(3.61e-06*250),
    MeVperADCPixel     = cms.double(3.61e-06),

    MisCalib_Mean      = cms.untracked.double(1.0),
    MisCalib_Sigma     = cms.untracked.double(0.00),

    UseCalibration  = cms.bool(True),
    calibrationPath = cms.string("file:Gains.root"),

    MaxNrStrips        = cms.untracked.uint32(255)
)

dedxSmi = dedxProd.clone()
dedxSmi.Formula = cms.untracked.uint32(2)

dedxASmi = dedxProd.clone()
dedxASmi.Formula = cms.untracked.uint32(3)


####################################################################################



HSCPTreeBuilder = cms.EDProducer("HSCPTreeBuilder",
   tracks                  = cms.InputTag("TrackRefitter"),
   dEdxDiscrim             = cms.VInputTag("dedxHarm2", "dedxNPHarm2", "dedxCHarm2", "dedxCNPHarm2", "dedxProd", "dedxSmi" ,"dedxASmi"),
   muons                   = cms.InputTag("muons"),
   muontiming              = cms.InputTag("muons"),


   minTrackMomentum   = cms.untracked.double  (0.0),
   minTrackTMomentum  = cms.untracked.double  (0.0),
   maxTrackMomentum   = cms.untracked.double  (9999),
   maxTrackTMomentum  = cms.untracked.double  (9999),
   minTrackEta        = cms.untracked.double  (0.0),
   maxTrackEta        = cms.untracked.double  (2.5),

   reccordVertexInfo  = cms.untracked.bool(True),
   reccordTrackInfo   = cms.untracked.bool(True),
   reccordMuonInfo    = cms.untracked.bool(True),
   reccordGenInfo     = cms.untracked.bool(False),
)

HSCPTreeBuilderSeq = cms.Sequence(dedxHarm2 + dedxCHarm2 + dedxNPHarm2 + dedxCNPHarm2 + dedxProd + dedxSmi + dedxASmi + HSCPTreeBuilder );






