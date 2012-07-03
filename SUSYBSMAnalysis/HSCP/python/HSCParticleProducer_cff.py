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
    MeVperADCStrip = cms.double(3.61e-06*265),
    MeVperADCPixel = cms.double(3.61e-06),

    MisCalib_Mean      = cms.untracked.double(1.0),
    MisCalib_Sigma     = cms.untracked.double(0.00),

    UseCalibration  = cms.bool(False),
    calibrationPath = cms.string(""),
    ShapeTest       = cms.bool(True),
)

dedxTru40 = cms.EDProducer("DeDxEstimatorProducer",
    tracks                     = cms.InputTag("TrackRefitter"),
    trajectoryTrackAssociation = cms.InputTag("TrackRefitter"),

    estimator      = cms.string('truncated'),
    fraction       = cms.double(0.4),

    UseStrip       = cms.bool(True),
    UsePixel       = cms.bool(True),
    MeVperADCStrip = cms.double(3.61e-06*265),
    MeVperADCPixel = cms.double(3.61e-06),

    MisCalib_Mean      = cms.untracked.double(1.0),
    MisCalib_Sigma     = cms.untracked.double(0.00),

    UseCalibration  = cms.bool(False),
    calibrationPath = cms.string(""),
    ShapeTest       = cms.bool(True),
)

dedxNPHarm2                  = dedxHarm2.clone()
dedxNPHarm2.UsePixel         = cms.bool(False)

dedxNPTru40                  = dedxTru40.clone()
dedxNPTru40.UsePixel         = cms.bool(False)

dedxNSHarm2                  = dedxHarm2.clone()
dedxNSHarm2.UseStrip         = cms.bool(False)

dedxNSTru40                  = dedxTru40.clone()
dedxNSTru40.UseStrip         = cms.bool(False)


dedxNSTHarm2                  = dedxHarm2.clone()
dedxNSTHarm2.ShapeTest        = cms.bool(False)



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
    MeVperADCStrip     = cms.double(3.61e-06*265),
    MeVperADCPixel     = cms.double(3.61e-06),

    MisCalib_Mean      = cms.untracked.double(1.0),
    MisCalib_Sigma     = cms.untracked.double(0.00),

    UseCalibration  = cms.bool(False),
    calibrationPath = cms.string("file:Gains.root"),
    ShapeTest          = cms.bool(True),

    MaxNrStrips        = cms.untracked.uint32(255)
)

dedxASmi = dedxProd.clone()
dedxASmi.Formula = cms.untracked.uint32(3)

dedxNPProd = dedxProd.clone()
dedxNPProd.UsePixel = cms.bool(False)

dedxNPASmi = dedxASmi.clone()
dedxNPASmi.UsePixel = cms.bool(False)


####################################################################################
#   HIT-DEDX Information
####################################################################################

dedxHitInfo               = cms.EDProducer("HSCPDeDxInfoProducer",
    tracks                     = cms.InputTag("TrackRefitter"),
    trajectoryTrackAssociation = cms.InputTag("TrackRefitter"),

    Reccord            = cms.untracked.string("SiStripDeDxMip_3D_Rcd"),
    Formula            = cms.untracked.uint32(0),
#    ProbabilityMode    = cms.untracked.string("Integral"),
    ProbabilityMode    = cms.untracked.string("Accumulation"),


    UseStrip           = cms.bool(True),
    UsePixel           = cms.bool(True),
    MeVperADCStrip     = cms.double(3.61e-06*265),
    MeVperADCPixel     = cms.double(3.61e-06),

    MisCalib_Mean      = cms.untracked.double(1.0),
    MisCalib_Sigma     = cms.untracked.double(0.00),

    UseCalibration  = cms.bool(False),
    calibrationPath = cms.string("file:Gains.root"),
    ShapeTest          = cms.bool(True),

    MaxNrStrips        = cms.untracked.uint32(255)
)


from RecoLocalMuon.DTSegment.dt4DSegments_MTPatternReco4D_LinearDriftFromDBLoose_cfi import *
dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.AlphaMaxPhi = 1.0
dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.AlphaMaxTheta = 0.9
dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.segmCleanerMode = 2
dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.MaxChi2 = 1.0
dt4DSegmentsMT = dt4DSegments.clone()
dt4DSegmentsMT.Reco4DAlgoConfig.recAlgoConfig.stepTwoFromDigi = True
dt4DSegmentsMT.Reco4DAlgoConfig.Reco2DAlgoConfig.recAlgoConfig.stepTwoFromDigi = True

####################################################################################
#   MUON TIMING
####################################################################################

from RecoMuon.MuonIdentification.muonTiming_cfi import *
muontiming.MuonCollection = cms.InputTag("muons")
muontiming.TimingFillerParameters.UseECAL=False
muontiming.TimingFillerParameters.DTTimingParameters.MatchParameters.DTsegments = "dt4DSegmentsMT"
muontiming.TimingFillerParameters.DTTimingParameters.HitsMin = 3
muontiming.TimingFillerParameters.DTTimingParameters.RequireBothProjections = False
muontiming.TimingFillerParameters.DTTimingParameters.DropTheta = True
muontiming.TimingFillerParameters.DTTimingParameters.DoWireCorr = True
muontiming.TimingFillerParameters.DTTimingParameters.MatchParameters.DTradius = 1.0
muontiming.TimingFillerParameters.DTTimingParameters.HitError = 3

####################################################################################
#   HSCParticle Producer
####################################################################################

#ALL THIS IS NEEDED BY ECAL BETA CALCULATOR (TrackAssociator)
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
from TrackingTools.TrackAssociator.default_cfi import * 
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
#remove to avoid geometry loader conflict --> geometry should be loaded from the main cfg
#from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
#from Geometry.CaloEventSetup.CaloGeometry_cff import *
#from Geometry.CaloEventSetup.CaloTopology_cfi import *
#from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
#from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
#from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
#from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
#from Geometry.DTGeometry.dtGeometry_cfi import *
#from Geometry.RPCGeometry.rpcGeometry_cfi import *
#from Geometry.CSCGeometry.cscGeometry_cfi import *
#from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *


from SUSYBSMAnalysis.HSCP.HSCPSelections_cff import *



HSCParticleProducer = cms.EDFilter("HSCParticleProducer",
   TrackAssociatorParameterBlock, #Needed for ECAL/Track Matching

   #DOES THE PRODUCER ACT AS AN EDFILTER?
   filter = cms.bool(True),

   #WHAT (BETA) INFORMATION TO COMPUTE
   useBetaFromTk      = cms.bool(True),
   useBetaFromMuon    = cms.bool(True),
   useBetaFromRpc     = cms.bool(True),
   useBetaFromEcal    = cms.bool(True),

   #TAG OF THE REQUIRED INPUT COLLECTION (ONLY ACTIVATED CALCULATOR)
   tracks             = cms.InputTag("TrackRefitter"),
   tracksIsolation    = cms.InputTag("generalTracks"),
   muons              = cms.InputTag("muons"),
   MTmuons            = cms.InputTag("RefitMTMuons"),
   EBRecHitCollection = cms.InputTag("ecalRecHit:EcalRecHitsEB"),
   EERecHitCollection = cms.InputTag("ecalRecHit:EcalRecHitsEE"),
   rpcRecHits         = cms.InputTag("rpcRecHits"),

   #TRACK SELECTION FOR THE HSCP SEED
   minMuP             = cms.double(25),
   minTkP             = cms.double(25),
   maxTkChi2          = cms.double(25),
   minTkHits          = cms.uint32(3),
   minSAMuPt          = cms.double(70),
   minMTMuPt          = cms.double(70),

   #MUON/TRACK MATCHING THRESHOLDS (ONLY IF NO MUON INNER TRACK)
   minDR              = cms.double(0.1),
   maxInvPtDiff       = cms.double(0.005),
   minMTDR              = cms.double(0.3),

   #SELECTION ON THE PRODUCED HSCP CANDIDATES (WILL STORE ONLY INTERESTING CANDIDATES)
   SelectionParameters = cms.VPSet(
      HSCPSelectionDefault,
      HSCPSelectionMTMuonOnly,
      HSCPSelectionSAMuonOnly,
   ),
)

####################################################################################
#   New Stand Alone Muon Producer
####################################################################################

from RecoMuon.Configuration.RecoMuon_cff import *
from RecoMuon.MuonSeedGenerator.ancientMuonSeed_cfi import *
MTancientMuonSeed = ancientMuonSeed.clone()
MTancientMuonSeed.DTRecSegmentLabel = "dt4DSegmentsMT"
NoRefitMTSAMuons = standAloneMuons.clone()
NoRefitMTSAMuons.InputObjects="MTancientMuonSeed"
RefitMTSAMuons=NoRefitMTSAMuons.clone()
RefitMTSAMuons.STATrajBuilderParameters.DoRefit=True

from RecoMuon.MuonIdentification.muons1stStep_cfi import *
RefitMTMuons = muons1stStep.clone()
RefitMTMuons.inputCollectionTypes = cms.vstring('outer tracks')
#RefitMTMuons.inputCollectionLabels = cms.VInputTag(cms.InputTag("RefitMTSAMuons",""))
RefitMTMuons.inputCollectionLabels = cms.VInputTag(cms.InputTag("RefitMTSAMuons","UpdatedAtVtx"))
RefitMTMuons.fillEnergy = False
RefitMTMuons.fillCaloCompatibility = False
RefitMTMuons.fillIsolation = False
RefitMTMuons.fillMatching = False

MuonSegmentProducer = cms.EDProducer("MuonSegmentProducer",
   CSCSegments        = cms.InputTag("cscSegments"),
   DTSegments         = cms.InputTag("dt4DSegmentsMT"),
)

MTmuontiming = muontiming.clone()
MTmuontiming.MuonCollection = "RefitMTMuons"

MuonOnlySeq = cms.Sequence(MTancientMuonSeed + NoRefitMTSAMuons + RefitMTSAMuons + RefitMTMuons + MuonSegmentProducer + MTmuontiming)

####################################################################################
#   HSCParticle Selector  (Just an Example of what we can do)
####################################################################################

HSCParticleSelector = cms.EDFilter("HSCParticleSelector",
   source = cms.InputTag("HSCParticleProducer"),
   filter = cms.bool(True),

   SelectionParameters = cms.VPSet(
      HSCPSelectionHighdEdx, #THE OR OF THE TWO SELECTION WILL BE APPLIED
      HSCPSelectionHighTOF,
   ),
)

####################################################################################
#   HSCP Candidate Sequence
####################################################################################

HSCParticleProducerSeq = cms.Sequence(offlineBeamSpot + TrackRefitter + dedxHarm2 + dedxTru40 + dedxNPHarm2 + dedxNPTru40 + dedxNSHarm2 + dedxNSTru40 + dedxProd + dedxASmi + dedxNPProd + dedxNPASmi + dedxNSTHarm2 + dedxHitInfo + dt4DSegmentsMT + muontiming + MuonOnlySeq + HSCParticleProducer)


