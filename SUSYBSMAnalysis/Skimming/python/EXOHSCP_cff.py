PT_CUT = 15. # P_t cut for tracks to match E/Hcal rechits. 

import FWCore.ParameterSet.Config as cms
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi

from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.TrackProducer.TrackRefitters_cff import *

generalTracksSkim = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = 'generalTracks',
    filter = True,
    applyBasicCuts = True,
    ptMin = 2.0,
    nHitMin = 3,
    chi2nMax = 10.,
)
TrackRefitter.TrajectoryInEvent = cms.bool(True)
TrackRefitter.src               = 'generalTracksSkim'

trackSkimmingAndRefit = cms.Sequence( generalTracksSkim)



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

dedxSeq = cms.Sequence(offlineBeamSpot + TrackRefitter + dedxHarm2 + dedxNPHarm2)



from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
from TrackingTools.TrackAssociator.default_cfi import *

muonEcalDetIds = cms.EDProducer("InterestingEcalDetIdProducer",
								inputCollection = cms.InputTag("muons")
								)
highPtTrackEcalDetIds = cms.EDProducer("HighPtTrackEcalDetIdProducer",
									   #TrackAssociatorParameterBlock
									   TrackAssociatorParameters=TrackAssociatorParameterBlock.TrackAssociatorParameters,
									   inputCollection = cms.InputTag("generalTracksSkim"),
									   TrackPt=cms.double(PT_CUT)
									   )



detIdProduceSeq = cms.Sequence(muonEcalDetIds+highPtTrackEcalDetIds)

reducedHSCPEcalRecHitsEB = cms.EDProducer("ReducedRecHitCollectionProducer",
     recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
     interestingDetIdCollections = cms.VInputTag(
	         #high p_t tracker track ids
	         cms.InputTag("highPtTrackEcalDetIds"),
             #muons
             cms.InputTag("muonEcalDetIds")
             ),
     reducedHitsCollection = cms.string('')
)
reducedHSCPEcalRecHitsEE = cms.EDProducer("ReducedRecHitCollectionProducer",
     recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
     interestingDetIdCollections = cms.VInputTag(
	         #high p_t tracker track ids
	         cms.InputTag("highPtTrackEcalDetIds"),
             #muons
             cms.InputTag("muonEcalDetIds")
             ),
     reducedHitsCollection = cms.string('')
)


ecalSeq = cms.Sequence(reducedHSCPEcalRecHitsEB+reducedHSCPEcalRecHitsEE)


reducedHSCPhbhereco = cms.EDProducer("ReduceHcalRecHitCollectionProducer",
									 recHitsLabel = cms.InputTag("hbhereco",""),
									 TrackAssociatorParameters=TrackAssociatorParameterBlock.TrackAssociatorParameters,
									 inputCollection = cms.InputTag("generalTracksSkim"),
									 TrackPt=cms.double(PT_CUT),					   
									 reducedHitsCollection = cms.string('')
)

hcalSeq = cms.Sequence(reducedHSCPhbhereco)


exoticaHSCPSeq = cms.Sequence( trackSkimmingAndRefit+detIdProduceSeq+ecalSeq+hcalSeq)





