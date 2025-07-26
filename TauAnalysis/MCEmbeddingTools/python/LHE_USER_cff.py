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
	--eventcontent TauEmbeddingCleaning \
	--datatier RAWRECO \
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