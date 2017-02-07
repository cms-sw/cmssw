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
globalPrevalidationTracking = cms.Sequence(
    simHitTPAssocProducer
  * tracksValidation
  * vertexValidation
)
globalPrevalidation = cms.Sequence(
    globalPrevalidationTracking
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


from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
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
#lite validation
globalValidationLiteTracking = cms.Sequence(globalValidation)

#lite pre-validation
globalPrevalidationLiteTracking = cms.Sequence(globalPrevalidation)
globalPrevalidationLiteTracking.replace(tracksValidation, tracksValidationLite)

from Validation.Configuration.gemSimValid_cff import *
from Validation.Configuration.me0SimValid_cff import *

baseCommonPreValidation = cms.Sequence(cms.SequencePlaceholder("mix"))
baseCommonValidation = cms.Sequence()

# Tracking-only validation
globalPrevalidationTrackingOnly = cms.Sequence(
      simHitTPAssocProducer
    + tracksValidationTrackingOnly
    + vertexValidationTrackingOnly
)
globalValidationTrackingOnly = cms.Sequence()


globalValidationJetMETonly = cms.Sequence(
                                   JetValidation 
                                 + METValidation
)

globalPrevalidationJetMETOnly = cms.Sequence(
				   jetPreValidSeq
				  +metPreValidSeq
)

globalPrevalidationHCAL = cms.Sequence()

globalValidationHCAL = cms.Sequence(
      hcalSimHitsValidationSequence
    + hcaldigisValidationSequence
    + hcalSimHitStudy
    + hcalRecHitsValidationSequence
    + calotowersValidationSequence
)

globalPrevalidationMuons = cms.Sequence(
      gemSimValid
    + me0SimValid
    + validSimHit
    + muondtdigianalyzer
    + cscDigiValidation
    + validationMuonRPCDigis
    + recoMuonValidation
    + rpcRecHitValidation_step
    + dtLocalRecoValidation_no2D
    + muonIdValDQMSeq
)

globalValidationMuons = cms.Sequence()

_run3_globalValidation = globalValidation.copy()
_run3_globalValidation += gemSimValid

_phase2_globalValidation = _run3_globalValidation.copy()
_phase2_globalValidation += me0SimValid

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toReplaceWith( globalValidation, _run3_globalValidation )
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith( globalValidation, _run3_globalValidation )
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith( globalValidation, _phase2_globalValidation )
