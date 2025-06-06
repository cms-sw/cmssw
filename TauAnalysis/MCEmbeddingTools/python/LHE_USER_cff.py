import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.tau_embedding_mu_to_mu_cff import tau_embedding_mu_to_mu
from Configuration.ProcessModifiers.tau_embedding_mu_to_e_cff import tau_embedding_mu_to_e
from Configuration.Eras.Modifier_run2_common_cff import run2_common
from Configuration.Eras.Modifier_run3_common_cff import run3_common
from RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi import bunchSpacingProducer

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