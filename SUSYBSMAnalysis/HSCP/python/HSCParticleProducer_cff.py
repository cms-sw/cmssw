import FWCore.ParameterSet.Config as cms

####################################################################################
#   BEAMSPOT + TRAJECTORY BUILDERS
####################################################################################

from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.TrackProducer.TrackRefitters_cff import *

####################################################################################
#   HIT-DEDX Information
####################################################################################

dedxHitInfo               = cms.EDProducer("HSCPDeDxInfoProducer",
    tracks                     = cms.InputTag("TrackRefitter"),
    trajectoryTrackAssociation = cms.InputTag("TrackRefitter"),

    Reccord            = cms.untracked.string("SiStripDeDxMip_3D_Rcd"),
    Formula            = cms.untracked.uint32(0),
    ProbabilityMode    = cms.untracked.string("Accumulation"),

    UseStrip           = cms.bool(True),
    UsePixel           = cms.bool(True),
    MeVperADCStrip     = cms.double(3.61e-06*265),
    MeVperADCPixel     = cms.double(3.61e-06),

    UseCalibration     = cms.bool(False),
    calibrationPath    = cms.string("file:Gains.root"),
    ShapeTest          = cms.bool(True),
)


####################################################################################
#   MUON TIMING
####################################################################################

from RecoMuon.MuonIdentification.muonTiming_cfi import *
muontiming.MuonCollection = cms.InputTag("muons")
muontiming.TimingFillerParameters.UseECAL=False

####################################################################################
#   HSCParticle Producer
####################################################################################

#ALL THIS IS NEEDED BY ECAL BETA CALCULATOR (TrackAssociator)
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
from TrackingTools.TrackAssociator.default_cfi import * 
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *

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
   MTmuons            = cms.InputTag("muons"),
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

HSCParticleProducerSeq = cms.Sequence(offlineBeamSpot + MeasurementTrackerEvent + TrackRefitter  + dedxHitInfo + muontiming + HSCParticleProducer)


