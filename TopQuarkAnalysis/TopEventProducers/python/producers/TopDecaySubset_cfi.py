import FWCore.ParameterSet.Config as cms

#
# module to make a persistent copy of all status 2
# (equivalent) genParticles of the top decay 
#
decaySubset = cms.EDProducer("TopDecaySubset",
    src       = cms.InputTag("genParticles"),
    ## restrict verbose printout to decay chains which
    ## contain certain particles (given by pdgId); 0
    ## means no restriction or selection
    pdgId     = cms.uint32(0),
    ## switch for type of generator listing
    ## 0: pythia   like
    ## 1: madgraph like
    genType   = cms.uint32(0)
)


