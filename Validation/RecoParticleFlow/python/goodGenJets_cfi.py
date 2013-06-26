import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import * 

# should do a cloning
genParticlesForJets.ignoreParticleIDs.append(14)
genParticlesForJets.ignoreParticleIDs.append(12)
genParticlesForJets.ignoreParticleIDs.append(16)
genParticlesForJets.excludeResonances = False

goodGenJets = cms.Sequence( 
    genJetParticles*
    recoGenJets
    )
