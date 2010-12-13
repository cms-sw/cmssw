import FWCore.ParameterSet.Config as cms
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi

from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.TrackProducer.TrackRefitters_cff import *

generalTracksSkim = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = 'generalTracks',
    filter = True,
    applyBasicCuts = True,
    ptMin = 10.0,
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
									   TrackPt=cms.double(20.0)
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
									 TrackPt=cms.double(20.0),					   
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




HSCPIsolation01 = cms.EDProducer("ProduceIsolationMap",
                                                                         inputCollection  = cms.InputTag("generalTracksSkim"),
                                                                         IsolationConeDR  = cms.double(0.1),
                                                                         TkIsolationPtCut = cms.double(10.0),
                                                                         TKLabel          = cms.InputTag("generalTracks"),
                                                                         HCALrecHitsLabel = cms.InputTag("hbhereco",""),
                                                                         EBrecHitsLabel   = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                                                                         EErecHitsLabel   = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                                                                         TrackAssociatorParameters=TrackAssociatorParameterBlock.TrackAssociatorParameters,
)

HSCPIsolation03 = HSCPIsolation01.clone()
HSCPIsolation03.IsolationConeDR  = cms.double(0.3)

HSCPIsolation05 = HSCPIsolation01.clone()
HSCPIsolation05.IsolationConeDR  = cms.double(0.5)

exoticaRecoIsoPhotonSeq = cms.EDFilter("MonoPhotonSkimmer",
  phoTag = cms.InputTag("photons::RECO"),
  selectEE = cms.bool(True),
  ecalisoOffsetEB = cms.double(4.2),
  ecalisoSlopeEB = cms.double(0.006),
  hcalisoOffsetEB = cms.double(2.2),
  hcalisoSlopeEB = cms.double(0.0025),
  hadoveremEB = cms.double(0.05),
  minPhoEtEB = cms.double(20.),
  trackIsoOffsetEB = cms.double(2.),
  trackIsoSlopeEB =  cms.double(0.001),
  etaWidthEB  = cms.double(0.013),
                                  
  ecalisoOffsetEE = cms.double(4.2),
  ecalisoSlopeEE = cms.double(0.006),
  hcalisoOffsetEE = cms.double(2.2),
  hcalisoSlopeEE = cms.double(0.0025),
  hadoveremEE = cms.double(0.05),
  minPhoEtEE = cms.double(20.),
  trackIsoOffsetEE = cms.double(2.),
  trackIsoSlopeEE =  cms.double(0.001),
  etaWidthEE  = cms.double(0.03),
                                  

 
)


exoticaHSCPSeq = cms.Sequence( trackerSeq+ecalSeq+hcalSeq+muonSeq+HSCPIsolation01+HSCPIsolation03+HSCPIsolation05)
exoticaHSCPIsoPhotonSeq = cms.Sequence(exoticaRecoIsoPhotonSeq + trackerSeq+ecalSeq+hcalSeq+muonSeq+HSCPIsolation01+HSCPIsolation03+HSCPIsolation05)





