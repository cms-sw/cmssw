from PhysicsTools.PatAlgos.slimming.genParticles_cff import *
from PhysicsTools.NanoAOD.genparticles_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *

hltJetPreValidSeq = cms.Sequence(
    prunedGenParticlesWithStatusOne
    + prunedGenParticles
    + finalGenParticles
    + genParticlesForJetsNoNu
    + ak4GenJetsNoNu
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toReplaceWith(jetPreValidSeq, hltJetPreValidSeq)
