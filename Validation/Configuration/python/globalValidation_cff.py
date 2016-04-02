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
#lite validation
globalValidationLiteTracking = cms.Sequence(globalValidation)

#lite pre-validation
globalPrevalidationLiteTracking = cms.Sequence(globalPrevalidation)
globalPrevalidationLiteTracking.replace(tracksValidation, tracksValidationLite)

# Tracking-only validation
globalPrevalidationTrackingOnly = cms.Sequence(
      simHitTPAssocProducer
    + tracksValidationTrackingOnly
    + vertexValidation
)
globalValidationTrackingOnly = cms.Sequence()

def _modifyGlobalValidationForPhase2( theProcess ):
    theProcess.load('Validation.Configuration.gemSimValid_cff')
    theProcess.load('Validation.Configuration.me0SimValid_cff')
    theProcess.globalValidation += theProcess.gemSimValid
    theProcess.globalValidation += theProcess.me0SimValid

from Configuration.StandardSequences.Eras import eras
modifyConfigurationStandardSequencesGlobalValidationForPhase2_ = eras.phase2_muon.makeProcessModifier( _modifyGlobalValidationForPhase2 )
