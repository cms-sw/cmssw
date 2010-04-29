import FWCore.ParameterSet.Config as cms


####################################################################################
#   BEAMSPOT + TRAJECTORY BUILDERS
####################################################################################

from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.TrackProducer.TrackRefitters_cff import *


####################################################################################
#   DEDX ESTIMATORS 
####################################################################################

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
#   DEDX DISCRIMINATORS 
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
#   MUON TIMING
####################################################################################

from RecoMuon.MuonIdentification.muonTiming_cfi import *
muontiming.MuonCollection = cms.InputTag("muons")

####################################################################################
#   HSCP TREE BUILDER
####################################################################################


HSCPTreeBuilder = cms.EDProducer("HSCPTreeBuilder",
   tracks                  = cms.InputTag("TrackRefitter"),
   dEdxDiscrim             = cms.VInputTag("dedxCHarm2", "dedxCNPHarm2", "dedxProd", "dedxASmi"),
   muons                   = cms.InputTag("muons"),
   muontiming              = cms.InputTag("muontiming"),

   reccordVertexInfo  = cms.untracked.bool(True),
   reccordTrackInfo   = cms.untracked.bool(True),
   reccordMuonInfo    = cms.untracked.bool(True),
   reccordGenInfo     = cms.untracked.bool(False),
)

####################################################################################
#   HSCP Tree Builder Sequence
####################################################################################

HSCPTreeBuilderSeq = cms.Sequence(offlineBeamSpot + TrackRefitter + dedxCHarm2 + dedxCNPHarm2 + dedxProd + dedxASmi + muontiming + HSCPTreeBuilder );
