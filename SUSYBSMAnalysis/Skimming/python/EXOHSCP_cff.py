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

trackerSeq = cms.Sequence( generalTracksSkim)


from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
from TrackingTools.TrackAssociator.default_cfi import *

muonEcalDetIds = cms.EDProducer("InterestingEcalDetIdProducer",
								inputCollection = cms.InputTag("muons")
								)
highPtTrackEcalDetIds = cms.EDProducer("HighPtTrackEcalDetIdProducer",
									   #TrackAssociatorParameterBlock
									   TrackAssociatorParameters=TrackAssociatorParameterBlock.TrackAssociatorParameters,
									   inputCollection = cms.InputTag("generalTracksSkim"),
									   TrackPt=cms.double(15.0)
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


ecalSeq = cms.Sequence(detIdProduceSeq+reducedHSCPEcalRecHitsEB+reducedHSCPEcalRecHitsEE)


reducedHSCPhbhereco = cms.EDProducer("ReduceHcalRecHitCollectionProducer",
									 recHitsLabel = cms.InputTag("hbhereco",""),
									 TrackAssociatorParameters=TrackAssociatorParameterBlock.TrackAssociatorParameters,
									 inputCollection = cms.InputTag("generalTracksSkim"),
									 TrackPt=cms.double(15.0),					   
									 reducedHitsCollection = cms.string('')
)

hcalSeq = cms.Sequence(reducedHSCPhbhereco)

muonsSkim = cms.EDProducer("UpdatedMuonInnerTrackRef",
    MuonTag        = cms.untracked.InputTag("muons"),
    OldTrackTag    = cms.untracked.InputTag("generalTracks"),
    NewTrackTag    = cms.untracked.InputTag("generalTracksSkim"),
    maxInvPtDiff   = cms.untracked.double(0.005),
    minDR          = cms.untracked.double(0.01),
)
muonSeq = cms.Sequence(muonsSkim)


exoticaHSCPSeq = cms.Sequence( trackerSeq+ecalSeq+hcalSeq+muonSeq)





