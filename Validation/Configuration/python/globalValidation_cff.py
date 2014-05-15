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
from Validation.RPCRecHits.rpcRecHitValidation_cfi import *
from Validation.DTRecHits.DTRecHitQuality_cfi import *
from Validation.RecoTau.DQMMCValidation_cfi import *
from DQMOffline.RecoB.dqmAnalyzer_cff import *

# filter/producer "pre-" sequence for globalValidation
globalPrevalidation = cms.Sequence( 
    simHitTPAssocProducer
  * tracksValidationSelectors
  * photonPrevalidationSequence
  * produceDenoms
  * prebTagSequence
)

# filter/producer "pre-" sequence for validation_preprod
preprodPrevalidation = cms.Sequence(
    tracksValidationSelectors
)

globalValidation = cms.Sequence(   trackerHitsValidation 
                                 + trackerDigisValidation 
                                 + trackerRecHitsValidation 
                                 + trackingTruthValid 
                                 + trackingRecHitsValid 
                                 + tracksValidation 
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
                                 + vertexValidation
                                 + egammaValidation
                                 + pfJetValidationSequence
                                 + pfMETValidationSequence
                                 + pfMuonValidationSequence
                                 + rpcRecHitValidation_step
				 + dtLocalRecoValidation_no2D
                                 + pfTauRunDQMValidation
                                 + bTagPlotsMCbcl
)
