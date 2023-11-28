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
from Validation.RecoParticleFlow.PFClusterValidation_cff import *
from Validation.RPCRecHits.rpcRecHitValidation_cfi import *
from Validation.DTRecHits.DTRecHitQuality_cfi import *
from Validation.CSCRecHits.cscRecHitValidation_cfi import *
from Validation.RecoTau.DQMMCValidation_cfi import *
from Validation.L1T.L1Validator_cfi import *
from Validation.SiPixelPhase1ConfigV.SiPixelPhase1OfflineDQM_sourceV_cff import *
from DQMOffline.RecoB.dqmAnalyzer_cff import *
from Validation.RecoB.BDHadronTrackValidation_cff import *
from Validation.Configuration.hgcalSimValid_cff import *
from Validation.Configuration.mtdSimValid_cff import *
from Validation.SiOuterTrackerV.OuterTrackerSourceConfigV_cff import *
from Validation.Configuration.ecalSimValid_cff import *
from Validation.SiTrackerPhase2V.Phase2TrackerValidationFirstStep_cff import *

# filter/producer "pre-" sequence for globalValidation
globalPrevalidationTracking = cms.Sequence(
    simHitTPAssocProducer
  * tracksValidation
  * vertexValidation
)
globalPrevalidation = cms.Sequence(
    globalPrevalidationTracking
  * photonPrevalidationSequence
  #* produceDenoms
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
                                 + pfClusterValidationSequence
                                 + rpcRecHitValidation_step
                                 + cscRecHitValidation
                                 + dtLocalRecoValidation_no2D
                                 + pfTauRunDQMValidation
                                 + bTagPlotsMCbcl
                                 + L1Validator
                                 + bdHadronTrackValidationSeq
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(globalValidation, globalValidation.copyAndExclude([
    # fastsim has no tracker digis and different tracker rechit and simhit structure => skipp
    trackerHitsValidation, trackerDigisValidation, trackerRecHitsValidation, trackingRecHitsValid,
    # the following depends on crossing frame of ecal simhits, which is a bit hard to implement in the fastsim workflow
    # besides: is this cross frame doing something, or is it a relic from the past?
    ecalDigisValidationSequence, ecalRecHitsValidationSequence
]))

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

# Pixel tracking only validation
globalPrevalidationPixelTrackingOnly = cms.Sequence(
      simHitTPAssocProducer
    + tracksValidationPixelTrackingOnly
    + vertexValidationPixelTrackingOnly
)
globalValidationPixelTrackingOnly = cms.Sequence()

globalValidationJetMETonly = cms.Sequence(
      JetValidation
    + METValidation
)

globalPrevalidationJetMETOnly = cms.Sequence(
      jetPreValidSeq
    + metPreValidSeq
)

# ECAL local reconstruction
globalPrevalidationECAL = cms.Sequence()
globalPrevalidationECALOnly = cms.Sequence(
      baseCommonPreValidation
    + globalPrevalidationECAL
)

globalValidationECAL = cms.Sequence(
      ecalSimHitsValidationSequence
    + ecalDigisValidationSequence
    + ecalRecHitsValidationSequence
    + ecalClustersValidationSequence
)
globalValidationECALOnly = cms.Sequence(
      ecalSimHitsValidationSequence
    + ecalDigisValidationSequence
    + ecalRecHitsValidationSequence
    + pfClusterCaloOnlyValidationSequence
)
from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
phase2_ecal_devel.toReplaceWith(ecalRecHitsValidationSequence, ecalRecHitsValidationSequencePhase2)

# HCAL local reconstruction
globalPrevalidationHCAL = cms.Sequence()

globalPrevalidationHCALOnly = cms.Sequence(
      baseCommonPreValidation
    + globalPrevalidationHCAL
)

hcalRecHitsOnlyValidationSequence = hcalRecHitsValidationSequence.copyAndExclude([NoiseRatesValidation])

globalValidationHCAL = cms.Sequence(
      hcalSimHitsValidationSequence
    + hcaldigisValidationSequence
    + hcalSimHitStudy
)

globalValidationHCALOnly = cms.Sequence(
      hcalSimHitsValidationSequence
    + hcaldigisValidationSequence
    + hcalSimHitStudy
    + hcalRecHitsOnlyValidationSequence
    + pfClusterCaloOnlyValidationSequence
)

globalValidationHGCal = cms.Sequence(hgcalValidation)
globalPrevalidationHGCal = cms.Sequence(hgcalAssociators, ticlSimTrackstersTask)

globalValidationMTD = cms.Sequence()

globalValidationOuterTracker = cms.Sequence(OuterTrackerSourceV)

globalPrevalidationMuons = cms.Sequence(
      gemSimValid
    + me0SimValid
    + validSimHit
    + muondtdigianalyzer
    + cscDigiValidation
    + validationMuonRPCDigis
    + recoMuonValidation
    + rpcRecHitValidation_step
    + cscRecHitValidation
    + dtLocalRecoValidation_no2D
    + muonIdValDQMSeq
)

globalValidationMuons = cms.Sequence()

_phase_1_globalValidation = globalValidation.copy()
_phase_1_globalValidation += siPixelPhase1OfflineDQM_sourceV

_phase_1_globalValidationPixelTrackingOnly =  globalValidationPixelTrackingOnly.copy()
_phase_1_globalValidationPixelTrackingOnly += siPixelPhase1ValidationPixelTrackingOnly_sourceV

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
(phase1Pixel & ~fastSim).toReplaceWith( globalValidation, _phase_1_globalValidation ) #module siPixelPhase1OfflineDQM_sourceV can't run in FastSim since siPixelClusters of type edmNew::DetSetVector are not produced
(phase1Pixel & ~fastSim).toReplaceWith( globalValidationPixelTrackingOnly, _phase_1_globalValidationPixelTrackingOnly ) #module siPixelPhase1OfflineDQM_sourceV can't run in FastSim since siPixelClusters of type edmNew::DetSetVector are not produced

_run3_globalValidation = globalValidation.copy()
_run3_globalValidation += gemSimValid

_phase2_globalValidation = _run3_globalValidation.copy()
_phase2_globalValidation += trackerphase2ValidationSource
_phase2_globalValidation += me0SimValid

_phase2_ge0_globalValidation = _run3_globalValidation.copy()
_phase2_ge0_globalValidation += trackerphase2ValidationSource

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toReplaceWith( globalValidation, _run3_globalValidation )
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith( globalValidation, _run3_globalValidation )
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith( globalValidation, _phase2_globalValidation )
from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0
phase2_GE0.toReplaceWith( globalValidation, _phase2_ge0_globalValidation )
phase2_GE0.toReplaceWith( globalPrevalidationMuons, globalPrevalidationMuons.copyAndExclude([me0SimValid]) )
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toReplaceWith(globalValidation, globalValidation.copyAndExclude([pfTauRunDQMValidation]))
from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
phase2_timing_layer.toReplaceWith(globalValidationMTD, cms.Sequence(mtdSimValid+mtdDigiValid+mtdRecoValid))
