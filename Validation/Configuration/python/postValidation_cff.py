import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.PostProcessor_cff import *
from Validation.RecoTrack.PostProcessorTracker_cfi import *
from Validation.MuonIsolation.PostProcessor_cff import *
from Validation.CaloTowers.CaloTowersPostProcessor_cff import *
from Validation.HcalHits.SimHitsPostProcessor_cff import *
from Validation.HcalDigis.HcalDigisPostProcessor_cff import *
from Validation.HcalRecHits.hcalRecHitsPostProcessor_cff import *
from Validation.EventGenerator.PostProcessor_cff import *
from Validation.RecoEgamma.photonPostProcessor_cff import *
from Validation.RecoEgamma.electronPostValidationSequence_cff import *
from Validation.RecoEgamma.electronPostValidationSequenceMiniAOD_cff import *
from Validation.RecoParticleFlow.PFValidationClient_cff import *
from Validation.RPCRecHits.postValidation_cfi import *
from Validation.RecoTau.DQMMCValidation_cfi import *
from Validation.RecoVertex.PostProcessorVertex_cff import *
from Validation.RecoMET.METPostProcessor_cff import *
from DQMOffline.RecoB.dqmCollector_cff import *


postValidationTracking = cms.Sequence(
      postProcessorTrackSequence
    + postProcessorVertexSequence
)
postValidation = cms.Sequence(
      recoMuonPostProcessors
    + postValidationTracking
    + MuIsoValPostProcessor
    + calotowersPostProcessor
    + hcalSimHitsPostProcessor
    + hcaldigisPostProcessor
    + hcalrechitsPostProcessor
    + electronPostValidationSequence + photonPostProcessor
    + pfJetClient + pfMETClient + pfJetResClient + pfElectronClient
    + rpcRecHitPostValidation_step
    + runTauEff + makeBetterPlots
    + bTagCollectorSequenceMCbcl
    + METPostProcessor
)
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel

postValidation_preprod = cms.Sequence(
    recoMuonPostProcessors
  + postProcessorTrackSequence
  + MuIsoValPostProcessor
)  


postValidation_fastsim = cms.Sequence(
      recoMuonPostProcessors
    + postProcessorTrackSequence
    + MuIsoValPostProcessor
    + photonPostProcessor
    + bTagCollectorSequenceMC
    + runTauEff
)

from Validation.MuonGEMHits.PostProcessor_cff import *
from Validation.MuonGEMDigis.PostProcessor_cff import *
from Validation.MuonGEMRecHits.PostProcessor_cff import *
from Validation.MuonME0Validation.PostProcessor_cff import *
from Validation.HGCalValidation.HGCalPostProcessor_cff import *

postValidation_common = cms.Sequence()

postValidation_trackingOnly = cms.Sequence(
      postProcessorTrackSequenceTrackingOnly
    + postProcessorVertexSequence
)

postValidation_muons = cms.Sequence(
    recoMuonPostProcessors
    + MuonGEMHitsPostProcessors
    + MuonGEMDigisPostProcessors
    + MuonGEMRecHitsPostProcessors
    + MuonME0DigisPostProcessors
    + MuonME0SegPostProcessors
    + rpcRecHitPostValidation_step
)

postValidation_JetMET = cms.Sequence(
    METPostProcessor
)

postValidation_HCAL = cms.Sequence(
      hcalSimHitsPostProcessor
    + hcaldigisPostProcessor
    + hcalrechitsPostProcessor
    + calotowersPostProcessor
)
 
postValidation_gen = cms.Sequence(
    EventGeneratorPostProcessor
)

postValidationCosmics = cms.Sequence(
    postProcessorMuonMultiTrack
)

postValidationMiniAOD = cms.Sequence(
    electronPostValidationSequenceMiniAOD
)

_run3_postValidation = postValidation.copy()
_run3_postValidation += MuonGEMHitsPostProcessors
_run3_postValidation += MuonGEMDigisPostProcessors
_run3_postValidation += MuonGEMRecHitsPostProcessors

_phase2_postValidation = _run3_postValidation.copy()
_phase2_postValidation += hgcalPostProcessor
_phase2_postValidation += MuonME0DigisPostProcessors
_phase2_postValidation += MuonME0SegPostProcessors

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toReplaceWith( postValidation, _run3_postValidation )
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith( postValidation, _run3_postValidation )
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith( postValidation, _phase2_postValidation )
