import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

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


postValidation = cms.Sequence(
      recoMuonPostProcessors
    + postProcessorTrackSequence
    + postProcessorVertexSequence
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
eras.phase1Pixel.toReplaceWith(postValidation, postValidation.copyAndExclude([ # FIXME
    runTauEff # Excessive printouts because 2017 doesn't have HLT yet
]))

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

postValidation_trackingOnly = cms.Sequence(
      postProcessorTrackSequenceTrackingOnly
    + postProcessorVertex
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

def _modifyPostValidationForRun3( theProcess ):
    theProcess.load('Validation.MuonGEMHits.PostProcessor_cff')
    theProcess.load('Validation.MuonGEMDigis.PostProcessor_cff')
    theProcess.load('Validation.MuonGEMRecHits.PostProcessor_cff')
    theProcess.load('Validation.HGCalValidation.HGCalPostProcessor_cff')
    theProcess.postValidation += theProcess.MuonGEMHitsPostProcessors
    theProcess.postValidation += theProcess.MuonGEMDigisPostProcessors
    theProcess.postValidation += theProcess.MuonGEMRecHitsPostProcessors
    theProcess.postValidation += theProcess.hgcalPostProcessor

from Configuration.StandardSequences.Eras import eras
modifyConfigurationStandardSequencesPostValidationForRun3_ = eras.run3_GEM.makeProcessModifier( _modifyPostValidationForRun3 )
