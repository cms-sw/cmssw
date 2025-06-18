"""
This config fragment generates LHE information for tau embedding. The selection step must be carried out beforehand.
It's normally used together with the cleaning step.
With `--procModifiers` one can specify wheather to simulate/embed muons (`tau_embedding_mu_to_mu`) or electrons (`tau_embedding_mu_to_e`) instead of taus.
To use this config fragment, a cmsDriver command like the following can be used:
```
cmsDriver.py \
	--step USER:TauAnalysis/MCEmbeddingTools/LHE_USER_cff.embeddingLHEProducerTask,RAW2DIGI,RECO:TauAnalysis/MCEmbeddingTools/Cleaning_RECO_cff.reconstruction \
	--processName LHEembeddingCLEAN \
	--data \
	--scenario pp \
	--eventcontent RAWRECO \
	--datatier RAWRECO \
	--outputCommands 'drop *_*_*_SELECT','drop recoIsoDepositedmValueMap_muIsoDepositTk_*_*','drop recoIsoDepositedmValueMap_muIsoDepositTkDisplaced_*_*','drop *_ctppsProtons_*_*','drop *_ctppsLocalTrackLiteProducer_*_*','drop *_ctppsDiamondLocalTracks_*_*','drop *_ctppsDiamondRecHits_*_*','drop *_ctppsDiamondRawToDigi_*_*','drop *_ctppsPixelLocalTracks_*_*','drop *_ctppsPixelRecHits_*_*','drop *_ctppsPixelClusters_*_*','drop *_ctppsPixelDigis_*_*','drop *_totemRPLocalTrackFitter_*_*','drop *_totemRPUVPatternFinder_*_*','drop *_totemRPRecHitProducer_*_*','drop *_totemRPClusterProducer_*_*','drop *_totemRPRawToDigi_*_*','drop *_muonSimClassifier_*_*','keep *_patMuonsAfterID_*_SELECT','keep *_slimmedMuons_*_SELECT','keep *_selectedMuonsForEmbedding_*_SELECT','keep recoVertexs_offlineSlimmedPrimaryVertices_*_SELECT','keep *_firstStepPrimaryVertices_*_SELECT','keep *_offlineBeamSpot_*_SELECT','keep *_l1extraParticles_*_SELECT','keep TrajectorySeeds_*_*_*','keep recoElectronSeeds_*_*_*','keep *_generalTracks_*_LHEembeddingCLEAN','keep *_generalTracks_*_CLEAN','keep *_cosmicsVetoTracksRaw_*_LHEembeddingCLEAN','keep *_cosmicsVetoTracksRaw_*_CLEAN','keep *_electronGsfTracks_*_LHEembeddingCLEAN','keep *_electronGsfTracks_*_CLEAN','keep *_lowPtGsfEleGsfTracks_*_LHEembeddingCLEAN','keep *_lowPtGsfEleGsfTracks_*_CLEAN','keep *_displacedTracks_*_LHEembeddingCLEAN','keep *_displacedTracks_*_CLEAN','keep *_ckfOutInTracksFromConversions_*_LHEembeddingCLEAN','keep *_ckfOutInTracksFromConversions_*_CLEAN','keep *_muons1stStep_*_LHEembeddingCLEAN','keep *_muons1stStep_*_CLEAN','keep *_displacedMuons1stStep_*_LHEembeddingCLEAN','keep *_displacedMuons1stStep_*_CLEAN','keep *_conversions_*_LHEembeddingCLEAN','keep *_conversions_*_CLEAN','keep *_allConversions_*_LHEembeddingCLEAN','keep *_allConversions_*_CLEAN','keep *_particleFlowTmp_*_LHEembeddingCLEAN','keep *_particleFlowTmp_*_CLEAN','keep *_ecalDigis_*_LHEembeddingCLEAN','keep *_ecalDigis_*_CLEAN','keep *_hcalDigis_*_LHEembeddingCLEAN','keep *_hcalDigis_*_CLEAN','keep *_ecalRecHit_*_LHEembeddingCLEAN','keep *_ecalRecHit_*_CLEAN','keep *_ecalPreshowerRecHit_*_LHEembeddingCLEAN','keep *_ecalPreshowerRecHit_*_CLEAN','keep *_hbhereco_*_LHEembeddingCLEAN','keep *_hbhereco_*_CLEAN','keep *_horeco_*_LHEembeddingCLEAN','keep *_horeco_*_CLEAN','keep *_hfreco_*_LHEembeddingCLEAN','keep *_hfreco_*_CLEAN','keep *_standAloneMuons_*_LHEembeddingCLEAN','keep *_glbTrackQual_*_LHEembeddingCLEAN','keep *_externalLHEProducer_*_LHEembedding','keep *_externalLHEProducer_*_LHEembeddingCLEAN' \
	--procModifiers tau_embedding_mu_to_mu \
    --era ... \
    --conditions ... \
    --filein ... \
    --fileout ...
```
"""
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run2_common_cff import run2_common
from Configuration.Eras.Modifier_run3_common_cff import run3_common
from Configuration.ProcessModifiers.tau_embedding_mu_to_e_cff import (
    tau_embedding_mu_to_e,
)
from Configuration.ProcessModifiers.tau_embedding_mu_to_mu_cff import (
    tau_embedding_mu_to_mu,
)
from RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi import bunchSpacingProducer

# As we want to exploit the toModify and toReplaceWith features of the FWCore/ParameterSet/python/Config.py Modifier class,
# we need a general modifier that is always applied.
# maybe this can also be replaced by a specific embedding process modifier
generalModifier = run2_common | run3_common

externalLHEProducer = cms.EDProducer("EmbeddingLHEProducer",
    src = cms.InputTag("selectedMuonsForEmbedding","",""),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices","","SELECT"),
    particleToEmbed = cms.int32(15),
)
# if running mu->mu embedding simulate muon (pid=13) instead of a tau (pid=15)
tau_embedding_mu_to_mu.toModify(externalLHEProducer, particleToEmbed = cms.int32(13))
# if running mu->e embedding simulate electron (pid=11) instead of a tau (pid=15)
tau_embedding_mu_to_e.toModify(externalLHEProducer, particleToEmbed = cms.int32(11))

generalModifier.toModify(bunchSpacingProducer, bunchSpacingOverride = cms.uint32(25), overrideBunchSpacing = cms.bool(True))


embeddingLHEProducerTask = cms.Sequence(externalLHEProducer)