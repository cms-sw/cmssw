"""
This config fragment generates LHE information for tau embedding. The selection step must be carried out beforehand.
It's normally used together with the cleaning step.
With `--procModifiers` one can specify wheather to simulate/embed muons (`tau_embedding_mu_to_mu`) or electrons (`tau_embedding_mu_to_e`) instead of taus.
To use this config fragment, a cmsDriver command like the following can be used:
```
cmsDriver.py \
	--step USER:TauAnalysis/MCEmbeddingTools/LHE_USER_cff.embeddingLHEProducerTask,RAW2DIGI,RECO \
	--processName LHEembeddingCLEAN \
	--data \
	--scenario pp \
	--eventcontent TauEmbeddingCleaning \
	--datatier RAWRECO \
	--procModifiers tau_embedding_cleaning,tau_embedding_mu_to_mu \
    --era ... \
    --conditions ... \
    --filein ... \
    --fileout ...
```
"""
import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.tau_embedding_mu_to_e_cff import (
    tau_embedding_mu_to_e,
)
from Configuration.ProcessModifiers.tau_embedding_mu_to_mu_cff import (
    tau_embedding_mu_to_mu,
)

externalLHEProducer = cms.EDProducer("EmbeddingLHEProducer",
    src = cms.InputTag("selectedMuonsForEmbedding","",""),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices","","SELECT"),
    particleToEmbed = cms.int32(15),
)
# if running mu->mu embedding simulate muon (pid=13) instead of a tau (pid=15)
tau_embedding_mu_to_mu.toModify(externalLHEProducer, particleToEmbed = cms.int32(13))
# if running mu->e embedding simulate electron (pid=11) instead of a tau (pid=15)
tau_embedding_mu_to_e.toModify(externalLHEProducer, particleToEmbed = cms.int32(11))

# switch on bunch spacing override to 25ns for tau embedding in
# RecoLuminosity/LumiProducer/python/bunchSpacingProducer_cfi.py


embeddingLHEProducerTask = cms.Sequence(externalLHEProducer)