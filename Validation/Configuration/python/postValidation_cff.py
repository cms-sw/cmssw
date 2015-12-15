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
from Validation.RecoParticleFlow.PFValidationClient_cff import *
from Validation.RPCRecHits.postValidation_cfi import *
from Validation.RecoTau.DQMMCValidation_cfi import *
from Validation.RecoVertex.PostProcessorVertex_cff import *
from Validation.RecoMET.METPostProcessor_cff import *
from DQMOffline.RecoB.dqmCollector_cff import *

from Configuration.StandardSequences.Eras import eras

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

postValidation_preprod = cms.Sequence(
    recoMuonPostProcessors
  + postProcessorTrackSequence
  + MuIsoValPostProcessor
)  

if eras.phase1Pixel.isChosen():
    # For starters, include only tracking validation
    # The rest should be added back once somebody checks that they
    # work, and those that do not, get fixed
    postValidation.remove(recoMuonPostProcessors)
    postValidation.remove(MuIsoValPostProcessor)
    postValidation.remove(calotowersPostProcessor)
    postValidation.remove(hcalSimHitsPostProcessor)
    postValidation.remove(hcaldigisPostProcessor)
    postValidation.remove(hcalrechitsPostProcessor)
    postValidation.remove(electronPostValidationSequence)
    postValidation.remove(photonPostProcessor)
    postValidation.remove(pfJetClient)
    postValidation.remove(pfMETClient)
    postValidation.remove(pfJetResClient)
    postValidation.remove(pfElectronClient)
    postValidation.remove(rpcRecHitPostValidation_step)
    postValidation.remove(runTauEff)
    postValidation.remove(makeBetterPlots)
    postValidation.remove(bTagCollectorSequenceMCbcl)
    postValidation.remove(METPostProcessor)
    postValidation_preprod.remove(recoMuonPostProcessors)
    postValidation_preprod.remove(MuIsoValPostProcessor)

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
