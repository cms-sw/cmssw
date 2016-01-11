import FWCore.ParameterSet.Config as cms

from SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi import *
from Validation.TrackerHits.trackerHitsValidation_cff import *
from Validation.TrackerDigis.trackerDigisValidation_cff import *
from Validation.TrackerRecHits.trackerRecHitsValidation_cff import *
from Validation.TrackingMCTruth.trackingTruthValidation_cfi import *
from Validation.RecoTrack.SiTrackingRecHitsValid_cff import *
from Validation.RecoTrack.TrackValidation_cff import *
from Validation.EcalHits.ecalSimHitsValidationSequence_cff import *
from Validation.EcalDigis.ecalDigisValidationSequence_cff import *
from Validation.EcalRecHits.ecalRecHitsValidationSequence_cff import *
from Validation.EcalClusters.ecalClustersValidationSequence_cff import *
from Validation.HcalHits.SimHitsValidationSequence_cff import *
from Validation.HcalDigis.hcalDigisValidationSequence_cff import *
from Validation.HcalHits.HcalSimHitStudy_cfi import *
from Validation.HcalRecHits.hcalRecHitsValidationSequence_cff import *
from Validation.CaloTowers.calotowersValidationSequence_cff import *
from Validation.MuonHits.muonHitsValidation_cfi import *
from Validation.MuonDTDigis.dtDigiValidation_cfi import *
from Validation.MuonCSCDigis.cscDigiValidation_cfi import *
from Validation.MuonRPCDigis.validationMuonRPCDigis_cfi import *
from Validation.RecoMuon.muonValidation_cff import *
from Validation.MuonIsolation.MuIsoVal_cff import *
from Validation.MuonIdentification.muonIdVal_cff import *
from Validation.Mixing.mixCollectionValidation_cfi import *
from Validation.RecoJets.JetValidation_cff import *
from Validation.RecoMET.METRelValForDQM_cff import *
from Validation.RecoVertex.VertexValidation_cff import *
from Validation.RecoEgamma.egammaValidation_cff import *
from Validation.RecoParticleFlow.PFJetValidation_cff  import *
from Validation.RecoParticleFlow.PFMETValidation_cff import *
from Validation.RecoParticleFlow.PFMuonValidation_cff import *
from Validation.RecoParticleFlow.PFElectronValidation_cff import *
from Validation.RecoParticleFlow.PFJetResValidation_cff import *
from Validation.RPCRecHits.rpcRecHitValidation_cfi import *
from Validation.DTRecHits.DTRecHitQuality_cfi import *
from Validation.RecoTau.DQMMCValidation_cfi import *
from Validation.L1T.L1Validator_cfi import *
from DQMOffline.RecoB.dqmAnalyzer_cff import *

# filter/producer "pre-" sequence for globalValidation
globalPrevalidation = cms.Sequence( 
    simHitTPAssocProducer
  * tracksValidation
  * vertexValidation
  * photonPrevalidationSequence
  * produceDenoms
  * prebTagSequenceMC
)

# filter/producer "pre-" sequence for validation_preprod
preprodPrevalidation = cms.Sequence(
    tracksPreValidation
)

globalValidation = cms.Sequence(   trackerHitsValidation 
                                 + trackerDigisValidation 
                                 + trackerRecHitsValidation 
                                 + trackingTruthValid 
                                 + trackingRecHitsValid 
                                 + ecalSimHitsValidationSequence 
                                 + ecalDigisValidationSequence 
                                 + ecalRecHitsValidationSequence 
                                 + ecalClustersValidationSequence
                                 + hcalSimHitsValidationSequence
                                 + hcaldigisValidationSequence
                                 + hcalSimHitStudy
                                 + hcalRecHitsValidationSequence
                                 + calotowersValidationSequence
                                 + validSimHit+muondtdigianalyzer 
                                 + cscDigiValidation
                                 + validationMuonRPCDigis 
                                 + recoMuonValidation 
                                 + muIsoVal_seq 
                                 + muonIdValDQMSeq 
                                 + mixCollectionValidation 
                                 + JetValidation 
                                 + METValidation
                                 + egammaValidation
                                 + pfJetValidationSequence
                                 + pfMETValidationSequence
                                 + pfElectronValidationSequence
                                 + pfJetResValidationSequence
                                 + pfMuonValidationSequence
                                 + rpcRecHitValidation_step
				 + dtLocalRecoValidation_no2D
                                 + pfTauRunDQMValidation
                                 + bTagPlotsMCbcl
                                 + L1Validator
)


from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    # fastsim has no tracker digis and different tracker rechit and simhit structure => skipp
    globalValidation.remove(trackerHitsValidation)
    globalValidation.remove(trackerDigisValidation)
    globalValidation.remove(trackerRecHitsValidation)
    globalValidation.remove(trackingRecHitsValid)
    # globalValidation.remove(mixCollectionValidation) # can be put back, once mixing is migrated to fastsim era
    # the following depends on crossing frame of ecal simhits, which is a bit hard to implement in the fastsim workflow
    # besides: is this cross frame doing something, or is it a relic from the past?
    globalValidation.remove(ecalDigisValidationSequence)
    globalValidation.remove(ecalRecHitsValidationSequence)
    
#lite tracking validator to be used in the Validation matrix
liteTrackValidator=trackValidator.clone()
liteTrackValidator.label=cms.VInputTag(cms.InputTag("generalTracks"),
                                          cms.InputTag("cutsRecoTracksHp")
                                          )

#lite validation
globalValidationLiteTracking = cms.Sequence(globalValidation)
globalValidationLiteTracking.replace(trackValidator,liteTrackValidator)

#lite pre-validation
# TODO: should we move this to TrackValidation_cff for better maintenance?
globalPrevalidationLiteTracking = cms.Sequence(globalPrevalidation)
globalPrevalidationLiteTracking.remove(cutsRecoTracksInitialStep)
globalPrevalidationLiteTracking.remove(cutsRecoTracksInitialStepHp)
globalPrevalidationLiteTracking.remove(cutsRecoTracksLowPtTripletStep)
globalPrevalidationLiteTracking.remove(cutsRecoTracksLowPtTripletStepHp)
globalPrevalidationLiteTracking.remove(cutsRecoTracksPixelPairStep)
globalPrevalidationLiteTracking.remove(cutsRecoTracksPixelPairStepHp)
globalPrevalidationLiteTracking.remove(cutsRecoTracksMixedTripletStep)
globalPrevalidationLiteTracking.remove(cutsRecoTracksMixedTripletStepHp)
globalPrevalidationLiteTracking.remove(cutsRecoTracksTobTecStep)
globalPrevalidationLiteTracking.remove(cutsRecoTracksTobTecStepHp)
globalPrevalidationLiteTracking.remove(cutsRecoTracksMuonSeededStepInOut)
globalPrevalidationLiteTracking.remove(cutsRecoTracksMuonSeededStepInOutHp)
globalPrevalidationLiteTracking.remove(cutsRecoTracksMuonSeededStepOutIn)
globalPrevalidationLiteTracking.remove(cutsRecoTracksMuonSeededStepOutInHp)
if eras.phase1Pixel.isChosen():
    globalPrevalidationLiteTracking.remove(cutsRecoTracksHighPtTripletStep)
    globalPrevalidationLiteTracking.remove(cutsRecoTracksHighPtTripletStepHp)
    globalPrevalidationLiteTracking.remove(cutsRecoTracksLowPtQuadStep)
    globalPrevalidationLiteTracking.remove(cutsRecoTracksLowPtQuadStepHp)
    globalPrevalidationLiteTracking.remove(cutsRecoTracksDetachedQuadStep)
    globalPrevalidationLiteTracking.remove(cutsRecoTracksDetachedQuadStepHp)
else:
    globalPrevalidationLiteTracking.remove(cutsRecoTracksDetachedTripletStep)
    globalPrevalidationLiteTracking.remove(cutsRecoTracksDetachedTripletStepHp)
    globalPrevalidationLiteTracking.remove(cutsRecoTracksPixelLessStep)
    globalPrevalidationLiteTracking.remove(cutsRecoTracksPixelLessStepHp)
    globalPrevalidationLiteTracking.remove(cutsRecoTracksJetCoreRegionalStep)
    globalPrevalidationLiteTracking.remove(cutsRecoTracksJetCoreRegionalStepHp)

# Tracking-only validation
globalPrevalidationTrackingOnly = cms.Sequence(
      simHitTPAssocProducer
    + tracksValidationTrackingOnly
    + vertexValidation
)
globalValidationTrackingOnly = cms.Sequence()
