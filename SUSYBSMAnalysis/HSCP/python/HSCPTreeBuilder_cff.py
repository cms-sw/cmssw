import FWCore.ParameterSet.Config as cms

####################################################################################
#   HSCP TREE BUILDER
####################################################################################

HSCPTreeBuilder = cms.EDProducer("HSCPTreeBuilder",
   HSCParticles       = cms.InputTag("HSCParticleProducer"),

   reccordVertexInfo  = cms.untracked.bool(True),
   reccordGenInfo     = cms.untracked.bool(False),
)
